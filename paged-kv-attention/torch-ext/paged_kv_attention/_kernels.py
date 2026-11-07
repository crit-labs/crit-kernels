"""The Triton kernels behind paged_kv_attention's functions (see __init__.py for what each computes).

Page kinds (KIND): 0 16-bit pages (BF16 or FP16), 1 FP8 E4M3 codes with one multiplier per layer for the keys and one
for the values, 2 FP8 E4M3 codes with an FP32 scale per position and KV head, 3 INT8 codes with an FP32 scale per
position and KV head. A position's slot is page * PAGE + offset, its keys at (slot * kv_heads + head) * dim.
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _e4m3_values(x):
    """E4M3 codes (uint8) to their FP32 values, exactly, with integer arithmetic (for GPUs without the conversion)."""
    b = x.to(tl.int32)
    e = (b >> 3) & 15
    m = (b & 7).to(tl.float32)
    unit = ((e + 117) << 23).to(tl.float32, bitcast=True)     # 2^(e - 10)
    mag = tl.where(e == 0, m * 0.001953125, (m + 8.0) * unit)
    return tl.where((b & 128) != 0, -mag, mag)


@triton.jit
def _e4m3_codes(x):
    """FP32 values within [-448, 448] to E4M3 codes (uint8), rounded to the nearest code with ties to even, as a
    float8_e4m3fn conversion rounds them."""
    a = tl.abs(x)
    bits = a.to(tl.int32, bitcast=True)
    ex = (bits >> 23) - 127
    # Normal codes: the mantissa rounded to 3 bits (exact scaling by a power of two, then rint: ties to even).
    frac = a * ((127 - ex) << 23).to(tl.float32, bitcast=True)
    mant = libdevice.rint((frac - 1.0) * 8.0).to(tl.int32)
    carry = mant == 8
    ex = tl.where(carry, ex + 1, ex)
    mant = tl.where(carry, 0, mant)
    normal = ((ex + 7) << 3) | mant
    # Below 2^-6 the codes are subnormal: multiples of 2^-9 (code 8 is the smallest normal value, 2^-6).
    sub = libdevice.rint(a * 512.0).to(tl.int32)
    code = tl.where(a < 0.015625, sub, normal)
    sign = (x.to(tl.int32, bitcast=True) >> 24) & 128
    return (code | sign).to(tl.uint8)


@triton.jit
def _values(x, KIND: tl.constexpr, HW_FP8: tl.constexpr, OUT: tl.constexpr):
    """Codes or 16-bit values as loaded from a page to the dot product's dtype, exactly (E4M3 codes load as bytes)."""
    if KIND == 1 or KIND == 2:
        if HW_FP8:
            y = x.to(tl.float8e4nv, bitcast=True).to(OUT)
        else:
            y = _e4m3_values(x).to(OUT)
    else:
        y = x.to(OUT)
    return y


# -- writes -----------------------------------------------------------------------------------------------------------
@triton.jit
def _write_kernel(k_ptr, v_ptr, slot_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, stride_kt, stride_kh, stride_vt, stride_vh,
                  k_div, v_div, HKV: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, BD: tl.constexpr,
                  KIND: tl.constexpr):
    """One token's KV head (axis 1: keys 0..HKV-1, values HKV..2HKV-1) into its slot; a slot below 0 is skipped."""
    t = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    d = tl.arange(0, BD)
    if slot >= 0:
        if head < HKV:
            h = head
            dm = d < D
            x = tl.load(k_ptr + t * stride_kt + h * stride_kh + d, mask=dm, other=0.0).to(tl.float32)
            dst = kp_ptr + (slot * HKV + h) * D + d
            sdst = ks_ptr + slot * HKV + h
            div = k_div
        else:
            h = head - HKV
            dm = d < DV
            x = tl.load(v_ptr + t * stride_vt + h * stride_vh + d, mask=dm, other=0.0).to(tl.float32)
            dst = vp_ptr + (slot * HKV + h) * DV + d
            sdst = vs_ptr + slot * HKV + h
            div = v_div
        if KIND == 0:
            tl.store(dst, x.to(dst.dtype.element_ty), mask=dm)
        elif KIND == 1:
            q = tl.minimum(tl.maximum(tl.div_rn(x, div), -448.0), 448.0)
            tl.store(dst, _e4m3_codes(q), mask=dm)
        elif KIND == 2:
            scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), axis=0), 448.0), 1e-30)
            q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
            tl.store(dst, _e4m3_codes(q), mask=dm)
            tl.store(sdst, scale)
        else:
            scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), axis=0), 127.0), 1e-30)
            code = tl.minimum(tl.maximum(libdevice.rint(tl.div_rn(x, scale)), -127.0), 127.0)
            tl.store(dst, code.to(tl.int8), mask=dm)
            tl.store(sdst, scale)


# -- decode and verify windows, split over KV partitions -------------------------------------------------------------
@triton.jit
def _ancestors(par_ptr, start, c, live, COLS: tl.constexpr):
    """Each column's ancestors and itself within its row's window, as a 64-bit mask (bit j: column j)."""
    one = tl.full(c.shape, 1, tl.int64)
    anc = tl.where(live, one << c, 0)
    par = tl.load(par_ptr + start + c, mask=live, other=-1).to(tl.int64)
    for _ in range(COLS):
        has = par >= 0
        at = tl.where(has, par, 0)
        anc = anc | tl.where(has, one << at, 0)
        par = tl.where(has, tl.load(par_ptr + start + at, mask=has, other=-1).to(tl.int64), -1)
    return anc


@triton.jit
def _window_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow_ptr, start_ptr, count_ptr, first_ptr, par_ptr,
                   po_ptr, pm_ptr, pl_ptr, stride_qt, stride_qh, tab_stride, tab_width, n_parts, sm_scale, k_mult,
                   v_mult, count_all, COLS: tl.constexpr, CPT: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
                   G: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, BLOCK_M: tl.constexpr, PIECE: tl.constexpr,
                   PAGE: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr, TREE: tl.constexpr,
                   COUNTS: tl.constexpr):
    """Program (column tile, partition b, row x KV head): the tile's query rows (its columns x the KV head's query
    heads) over pieces b, b + n_parts, ... of the row's keys, up to its last column's position; writes each row's
    unnormalized output, running maximum and sum into the partials."""
    mt = tl.program_id(0)
    b = tl.program_id(1)
    rk = tl.program_id(2)
    r = rk // HKV
    kvh = rk % HKV
    if COUNTS:
        count = tl.load(count_ptr + r)
    else:
        count = count_all
    c0 = mt * CPT
    first = tl.load(first_ptr + r).to(tl.int64)
    n_keys = first + tl.minimum(count, c0 + CPT)
    n_pieces = tl.cdiv(n_keys, PIECE).to(tl.int32)
    if (c0 < count) & (b < n_pieces):
        start = tl.load(start_ptr + r).to(tl.int64)
        trow = tl.load(trow_ptr + r).to(tl.int64)
        j = tl.arange(0, BLOCK_M)
        c = (c0 + j // G).to(tl.int64)
        g = j % G
        live = (j < CPT * G) & (c < count)
        d = tl.arange(0, D)
        dv = tl.arange(0, DV)
        q = tl.load(q_ptr + (start + c)[:, None] * stride_qt + (kvh * G + g)[:, None] * stride_qh + d[None, :],
                    mask=live[:, None], other=0.0)
        lim = first + c
        if TREE:
            anc = _ancestors(par_ptr, start, c, live, COLS)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, DV], tl.float32)
        ones = tl.full([PIECE, 16], 1.0, q.dtype)
        for i in range(b, n_pieces, n_parts):
            n = i * PIECE + tl.arange(0, PIECE)
            nm = n < n_keys
            pg = tl.load(tab_ptr + trow * tab_stride + tl.minimum(n // PAGE, tab_width - 1), mask=nm, other=0)
            slot = pg.to(tl.int64) * PAGE + n % PAGE
            kc = tl.load(k_ptr + (slot * HKV + kvh)[:, None] * D + d[None, :], mask=nm[:, None], other=0)
            s = tl.dot(q, tl.trans(_values(kc, KIND, HW_FP8, q.dtype)))
            if KIND >= 2:
                ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                s = s * (ksc * sm_scale)[None, :]
            else:
                s = s * (k_mult * sm_scale)
            if TREE:
                w = n - first
                inw = (w >= 0) & (w < count)
                bit = (anc[:, None] >> tl.where(inw, w, 0)[None, :]) & 1
                vis = nm[None, :] & ((n < first)[None, :] | (inw[None, :] & (bit != 0)))
            else:
                vis = nm[None, :] & (n[None, :] <= lim[:, None])
            s = tl.where(vis, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_sub = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_sub[:, None])
            alpha = tl.exp(m_i - m_sub)
            vc = tl.load(v_ptr + (slot * HKV + kvh)[:, None] * DV + dv[None, :], mask=nm[:, None], other=0)
            if KIND >= 2:
                vsc = tl.load(vs_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                pv = (p * vsc[None, :]).to(q.dtype)
            else:
                pv = (p * v_mult).to(q.dtype)
            acc = acc * alpha[:, None] + tl.dot(pv, _values(vc, KIND, HW_FP8, q.dtype))
            # The row sums on the tensor cores too: a fixed summation order, whatever the tile's layout.
            l_i = l_i * alpha + tl.max(tl.dot(p.to(q.dtype), ones), axis=1)
            m_i = m_new
        idx = ((r * COLS + c) * HQ + kvh * G + g) * n_parts + b
        tl.store(po_ptr + idx[:, None] * DV + dv[None, :], acc, mask=live[:, None])
        tl.store(pm_ptr + idx, m_i, mask=live)
        tl.store(pl_ptr + idx, l_i, mask=live)


@triton.jit
def _sparse_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow_ptr, start_ptr, count_ptr, first_ptr, par_ptr,
                   blk_ptr, blk_stride, n_blocks, block_size, po_ptr, pm_ptr, pl_ptr, stride_qt, stride_qh, tab_stride,
                   tab_width, n_parts, sm_scale, k_mult, v_mult, count_all, COLS: tl.constexpr, HQ: tl.constexpr,
                   HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, BLOCK_M: tl.constexpr,
                   PIECE: tl.constexpr, PAGE: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr,
                   TREE: tl.constexpr, COUNTS: tl.constexpr):
    """Program (partition b, KV head, row x column): one token's query heads of the KV head over the positions of its
    listed KV blocks (blocks[token][0 .. n_blocks - 1], -1 for none), read as one run of n_blocks * block_size
    positions in list order, its tiles b, b + n_parts, ...; positions past the token's own (or outside its tree) are
    not seen. Writes partials for every b below the tile count, empty ones included."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    rc = tl.program_id(2)
    r = rc // COLS
    c = (rc % COLS).to(tl.int64)
    if COUNTS:
        count = tl.load(count_ptr + r)
    else:
        count = count_all
    n_tiles = tl.cdiv(n_blocks * block_size, PIECE)
    if (c < count) & (b < n_tiles):
        start = tl.load(start_ptr + r).to(tl.int64)
        first = tl.load(first_ptr + r).to(tl.int64)
        trow = tl.load(trow_ptr + r).to(tl.int64)
        tok = start + c
        g = tl.arange(0, BLOCK_M)
        live = g < G
        d = tl.arange(0, D)
        dv = tl.arange(0, DV)
        q = tl.load(q_ptr + tok * stride_qt + (kvh * G + g)[:, None] * stride_qh + d[None, :], mask=live[:, None],
                    other=0.0)
        if TREE:
            # Every row is the same column: its mask (identical on every row) taken as a scalar.
            anc = tl.max(_ancestors(par_ptr, start, tl.zeros([BLOCK_M], tl.int64) + c, g >= 0, COLS), axis=0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, DV], tl.float32)
        ones = tl.full([PIECE, 16], 1.0, q.dtype)
        for i in range(b, n_tiles, n_parts):
            u = i * PIECE + tl.arange(0, PIECE)
            e = u // block_size
            blk = tl.load(blk_ptr + tok * blk_stride + e, mask=e < n_blocks, other=-1).to(tl.int64)
            n = blk * block_size + u % block_size
            if TREE:
                w = n - first
                inw = (w >= 0) & (w < count)
                seen = (n < first) | (inw & (((anc >> tl.where(inw, w, 0)) & 1) != 0))
            else:
                seen = n <= first + c
            nm = (blk >= 0) & (n < first + count) & seen
            pg = tl.load(tab_ptr + trow * tab_stride + tl.minimum(n // PAGE, tab_width - 1), mask=nm, other=0)
            slot = pg.to(tl.int64) * PAGE + n % PAGE
            kc = tl.load(k_ptr + (slot * HKV + kvh)[:, None] * D + d[None, :], mask=nm[:, None], other=0)
            s = tl.dot(q, tl.trans(_values(kc, KIND, HW_FP8, q.dtype)))
            if KIND >= 2:
                ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                s = s * (ksc * sm_scale)[None, :]
            else:
                s = s * (k_mult * sm_scale)
            s = tl.where(nm[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_sub = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_sub[:, None])
            alpha = tl.exp(m_i - m_sub)
            vc = tl.load(v_ptr + (slot * HKV + kvh)[:, None] * DV + dv[None, :], mask=nm[:, None], other=0)
            if KIND >= 2:
                vsc = tl.load(vs_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                pv = (p * vsc[None, :]).to(q.dtype)
            else:
                pv = (p * v_mult).to(q.dtype)
            acc = acc * alpha[:, None] + tl.dot(pv, _values(vc, KIND, HW_FP8, q.dtype))
            l_i = l_i * alpha + tl.max(tl.dot(p.to(q.dtype), ones), axis=1)
            m_i = m_new
        idx = ((r * COLS + c) * HQ + kvh * G + g) * n_parts + b
        tl.store(po_ptr + idx[:, None] * DV + dv[None, :], acc, mask=live[:, None])
        tl.store(pm_ptr + idx, m_i, mask=live)
        tl.store(pl_ptr + idx, l_i, mask=live)


@triton.jit
def _merge_kernel(po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr, start_ptr, count_ptr, first_ptr, n_parts, n_fixed,
                  stride_ot, stride_oh, count_all, COLS: tl.constexpr, HQ: tl.constexpr, DV: tl.constexpr,
                  PIECE: tl.constexpr, COUNTS: tl.constexpr, LSE: tl.constexpr):
    """Program (column, query head, row): the partials of the partitions that hold the column's keys, merged in
    partition order by their maxima and sums (log-sum-exp weights), into the output row (and its log-sum-exp)."""
    c = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.program_id(2)
    if COUNTS:
        count = tl.load(count_ptr + r)
    else:
        count = count_all
    if c < count:
        start = tl.load(start_ptr + r).to(tl.int64)
        if n_fixed > 0:
            nb = tl.minimum(n_parts, n_fixed)
        else:
            first = tl.load(first_ptr + r).to(tl.int64)
            nb = tl.minimum(n_parts, tl.cdiv(first + c + 1, PIECE)).to(tl.int32)
        base = ((r * COLS + c) * HQ + h).to(tl.int64) * n_parts
        dv = tl.arange(0, DV)
        top = tl.load(pm_ptr + base)
        for b in range(1, nb):
            top = tl.maximum(top, tl.load(pm_ptr + base + b))
        den = 0.0
        acc = tl.zeros([DV], tl.float32)
        for b in range(0, nb):
            mb = tl.load(pm_ptr + base + b)
            w = tl.where(mb == float("-inf"), 0.0, tl.exp(mb - top))
            den = den + w * tl.load(pl_ptr + base + b)
            acc = acc + w * tl.load(po_ptr + (base + b) * DV + dv)
        o = tl.where(den > 0, acc / den, 0.0)
        tl.store(out_ptr + (start + c) * stride_ot + h * stride_oh + dv, o.to(out_ptr.dtype.element_ty))
        if LSE:
            tl.store(lse_ptr + (start + c) * HQ + h, tl.where(den > 0, top + tl.log(den), float("-inf")))


# -- prompt chunks ----------------------------------------------------------------------------------------------------
@triton.jit
def _prefill_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, out_ptr, lse_ptr, T, pos0, sm_scale, k_mult, v_mult,
                    stride_qt, stride_qh, stride_ot, stride_oh, HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
                    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PAGE: tl.constexpr,
                    KIND: tl.constexpr, HW_FP8: tl.constexpr, INT8_QK: tl.constexpr, LSE: tl.constexpr):
    """Program (query block, query head): a chunk of one sequence's queries at positions pos0.. over its positions
    [0, own], causal, through its page table."""
    mb = tl.program_id(0)
    head = tl.program_id(1)
    kvh = head // G
    rows = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < T
    d = tl.arange(0, D)
    q = tl.load(q_ptr + rows[:, None] * stride_qt + head * stride_qh + d[None, :], mask=rmask[:, None], other=0.0)
    if INT8_QK:
        # Each query row to INT8 at its own scale (amax / 127, rounded half away from zero), in registers.
        qf = q.to(tl.float32)
        qs = tl.maximum(tl.max(tl.abs(qf), axis=1) / 127.0, 1e-30)
        qf = qf / qs[:, None]
        qi = tl.where(qf >= 0, qf + 0.5, qf - 0.5).to(tl.int8)
        qs = qs * sm_scale
    qpos = pos0 + rows
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    end = tl.minimum(pos0 + (mb + 1) * BLOCK_M, pos0 + T)
    for n0 in range(0, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < end
        pg = tl.load(tab_ptr + n // PAGE, mask=nm, other=0)
        slot = pg.to(tl.int64) * PAGE + n % PAGE
        kc = tl.load(k_ptr + (slot * HKV + kvh)[:, None] * D + d[None, :], mask=nm[:, None], other=0)
        if INT8_QK:
            ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
            s = tl.dot(qi, tl.trans(kc)).to(tl.float32) * qs[:, None] * ksc[None, :]
        elif KIND >= 2:
            ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
            s = tl.dot(q, tl.trans(_values(kc, KIND, HW_FP8, q.dtype))) * (ksc * sm_scale)[None, :]
        else:
            s = tl.dot(q, tl.trans(_values(kc, KIND, HW_FP8, q.dtype))) * (k_mult * sm_scale)
        s = tl.where((n[None, :] <= qpos[:, None]) & nm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_sub = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_sub[:, None])
        alpha = tl.exp(m_i - m_sub)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vc = tl.load(v_ptr + (slot * HKV + kvh)[:, None] * D + d[None, :], mask=nm[:, None], other=0)
        if KIND >= 2:
            vsc = tl.load(vs_ptr + slot * HKV + kvh, mask=nm, other=0.0)
            pv = (p * vsc[None, :]).to(q.dtype)
        else:
            pv = (p * v_mult).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(pv, _values(vc, KIND, HW_FP8, q.dtype))
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(out_ptr + rows[:, None] * stride_ot + head * stride_oh + d[None, :], o.to(out_ptr.dtype.element_ty),
             mask=rmask[:, None])
    if LSE:
        tl.store(lse_ptr + rows * HQ + head, m_i + tl.log(l_i), mask=rmask)
