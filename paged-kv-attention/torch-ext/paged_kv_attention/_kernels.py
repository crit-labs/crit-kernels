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
def _piece_scores(qrow, live, krow, nm, BLOCK_M: tl.constexpr, PIECE: tl.constexpr, D: tl.constexpr,
                  DC: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr):
    """The query rows' dot products with a piece's keys, the head dimension in chunks of DC (the chained dots keep a
    full dot's k-step order, so the bits are those of one dot over D; the chunks keep the tiles out of the spills)."""
    dc = tl.arange(0, DC)
    s = tl.zeros([BLOCK_M, PIECE], tl.float32)
    for ci in tl.static_range(D // DC):
        qc = tl.load(qrow[:, None] + (ci * DC + dc)[None, :], mask=live[:, None], other=0.0)
        kc = tl.load(krow[:, None] + (ci * DC + dc)[None, :], mask=nm[:, None], other=0)
        s = tl.dot(qc, tl.trans(_values(kc, KIND, HW_FP8, qc.dtype)), s)
    return s


@triton.jit
def _piece_values(acc0, acc1, acc2, acc3, alpha, pv, vrow, nm, DV: tl.constexpr, DC: tl.constexpr,
                  KIND: tl.constexpr, HW_FP8: tl.constexpr):
    """The value accumulators (up to four chunks of DC of the value dimension) rescaled by alpha and given a piece's
    probabilities times its values."""
    dc = tl.arange(0, DC)
    v0 = tl.load(vrow[:, None] + dc[None, :], mask=nm[:, None], other=0)
    acc0 = acc0 * alpha[:, None] + tl.dot(pv, _values(v0, KIND, HW_FP8, pv.dtype))
    if DV // DC > 1:
        v1 = tl.load(vrow[:, None] + (DC + dc)[None, :], mask=nm[:, None], other=0)
        acc1 = acc1 * alpha[:, None] + tl.dot(pv, _values(v1, KIND, HW_FP8, pv.dtype))
    if DV // DC > 2:
        v2 = tl.load(vrow[:, None] + (2 * DC + dc)[None, :], mask=nm[:, None], other=0)
        acc2 = acc2 * alpha[:, None] + tl.dot(pv, _values(v2, KIND, HW_FP8, pv.dtype))
    if DV // DC > 3:
        v3 = tl.load(vrow[:, None] + (3 * DC + dc)[None, :], mask=nm[:, None], other=0)
        acc3 = acc3 * alpha[:, None] + tl.dot(pv, _values(v3, KIND, HW_FP8, pv.dtype))
    return acc0, acc1, acc2, acc3


@triton.jit
def _partial_row(off_ptr, r, COLS: tl.constexpr, OFFSETS: tl.constexpr):
    """Row r's first partial row in its launch's partials: with OFFSETS its offset past the launch's first row's (the
    rows' tokens packed one after another), else r * COLS."""
    if OFFSETS:
        return (tl.load(off_ptr + r) - tl.load(off_ptr)).to(tl.int64)
    return r.to(tl.int64) * COLS


@triton.jit
def _store_partials(po_ptr, idx, live, acc0, acc1, acc2, acc3, DV: tl.constexpr, DC: tl.constexpr):
    dc = tl.arange(0, DC)
    row = po_ptr + idx[:, None] * DV
    tl.store(row + dc[None, :], acc0, mask=live[:, None])
    if DV // DC > 1:
        tl.store(row + (DC + dc)[None, :], acc1, mask=live[:, None])
    if DV // DC > 2:
        tl.store(row + (2 * DC + dc)[None, :], acc2, mask=live[:, None])
    if DV // DC > 3:
        tl.store(row + (3 * DC + dc)[None, :], acc3, mask=live[:, None])


@triton.jit
def _window_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow_ptr, start_ptr, count_ptr, first_ptr, par_ptr,
                   off_ptr, po_ptr, pm_ptr, pl_ptr, stride_qt, stride_qh, tab_stride, tab_width, n_parts, sm_scale,
                   k_mult, v_mult, count_all, COLS: tl.constexpr, CPT: tl.constexpr, HQ: tl.constexpr,
                   HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, BLOCK_M: tl.constexpr,
                   PIECE: tl.constexpr, PAGE: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr,
                   TREE: tl.constexpr, COUNTS: tl.constexpr, OFFSETS: tl.constexpr, DC: tl.constexpr):
    """Program (column tile, partition b, row x KV head): the tile's query rows (its columns x the KV head's query
    heads) over pieces b, b + n_parts, ... of the row's keys, up to its last column's position; writes each row's
    unnormalized output, running maximum and sum into the partials."""
    mt = tl.program_id(0)
    b = tl.program_id(1)
    rk = tl.program_id(2)
    r = rk // HKV
    kvh = rk % HKV
    if COUNTS:
        count = tl.minimum(tl.load(count_ptr + r), COLS)
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
        qrow = q_ptr + (start + c) * stride_qt + (kvh * G + g) * stride_qh
        lim = first + c
        if TREE:
            anc = _ancestors(par_ptr, start, c, live, COLS)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc0 = tl.zeros([BLOCK_M, DC], tl.float32)
        acc1 = tl.zeros([BLOCK_M, DC], tl.float32)
        acc2 = tl.zeros([BLOCK_M, DC], tl.float32)
        acc3 = tl.zeros([BLOCK_M, DC], tl.float32)
        ones = tl.full([PIECE, 16], 1.0, q_ptr.dtype.element_ty)
        for i in range(b, n_pieces, n_parts):
            n = i * PIECE + tl.arange(0, PIECE)
            nm = n < n_keys
            pg = tl.load(tab_ptr + trow * tab_stride + tl.minimum(n // PAGE, tab_width - 1), mask=nm, other=0)
            slot = pg.to(tl.int64) * PAGE + n % PAGE
            s = _piece_scores(qrow, live, k_ptr + (slot * HKV + kvh) * D, nm, BLOCK_M, PIECE, D, DC, KIND, HW_FP8)
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
            if KIND >= 2:
                vsc = tl.load(vs_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                pv = (p * vsc[None, :]).to(ones.dtype)
            else:
                pv = (p * v_mult).to(ones.dtype)
            acc0, acc1, acc2, acc3 = _piece_values(acc0, acc1, acc2, acc3, alpha, pv, v_ptr + (slot * HKV + kvh) * DV,
                                                   nm, DV, DC, KIND, HW_FP8)
            # The row sums on the tensor cores too: a fixed summation order, whatever the tile's layout.
            l_i = l_i * alpha + tl.max(tl.dot(p.to(ones.dtype), ones), axis=1)
            m_i = m_new
        idx = ((_partial_row(off_ptr, r, COLS, OFFSETS) + c) * HQ + kvh * G + g) * n_parts + b
        _store_partials(po_ptr, idx, live, acc0, acc1, acc2, acc3, DV, DC)
        tl.store(pm_ptr + idx, m_i, mask=live)
        tl.store(pl_ptr + idx, l_i, mask=live)


@triton.jit
def _sparse_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow_ptr, start_ptr, count_ptr, first_ptr, par_ptr,
                   off_ptr, blk_ptr, blk_stride, n_blocks, block_size, po_ptr, pm_ptr, pl_ptr, stride_qt, stride_qh,
                   tab_stride, tab_width, n_parts, sm_scale, k_mult, v_mult, count_all, COLS: tl.constexpr,
                   HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
                   BLOCK_M: tl.constexpr, PIECE: tl.constexpr, PAGE: tl.constexpr, KIND: tl.constexpr,
                   HW_FP8: tl.constexpr, TREE: tl.constexpr, COUNTS: tl.constexpr, OFFSETS: tl.constexpr,
                   DC: tl.constexpr):
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
        count = tl.minimum(tl.load(count_ptr + r), COLS)
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
        qrow = q_ptr + tok * stride_qt + (kvh * G + g) * stride_qh
        if TREE:
            # Every row is the same column: its mask (identical on every row) taken as a scalar.
            anc = tl.max(_ancestors(par_ptr, start, tl.zeros([BLOCK_M], tl.int64) + c, g >= 0, COLS), axis=0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc0 = tl.zeros([BLOCK_M, DC], tl.float32)
        acc1 = tl.zeros([BLOCK_M, DC], tl.float32)
        acc2 = tl.zeros([BLOCK_M, DC], tl.float32)
        acc3 = tl.zeros([BLOCK_M, DC], tl.float32)
        ones = tl.full([PIECE, 16], 1.0, q_ptr.dtype.element_ty)
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
            s = _piece_scores(qrow, live, k_ptr + (slot * HKV + kvh) * D, nm, BLOCK_M, PIECE, D, DC, KIND, HW_FP8)
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
            if KIND >= 2:
                vsc = tl.load(vs_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                pv = (p * vsc[None, :]).to(ones.dtype)
            else:
                pv = (p * v_mult).to(ones.dtype)
            acc0, acc1, acc2, acc3 = _piece_values(acc0, acc1, acc2, acc3, alpha, pv, v_ptr + (slot * HKV + kvh) * DV,
                                                   nm, DV, DC, KIND, HW_FP8)
            l_i = l_i * alpha + tl.max(tl.dot(p.to(ones.dtype), ones), axis=1)
            m_i = m_new
        idx = ((_partial_row(off_ptr, r, COLS, OFFSETS) + c) * HQ + kvh * G + g) * n_parts + b
        _store_partials(po_ptr, idx, live, acc0, acc1, acc2, acc3, DV, DC)
        tl.store(pm_ptr + idx, m_i, mask=live)
        tl.store(pl_ptr + idx, l_i, mask=live)


@triton.jit
def _merge_kernel(po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr, start_ptr, count_ptr, first_ptr, off_ptr, n_parts, n_fixed,
                  stride_ot, stride_oh, count_all, COLS: tl.constexpr, HQ: tl.constexpr, DV: tl.constexpr,
                  PIECE: tl.constexpr, COUNTS: tl.constexpr, OFFSETS: tl.constexpr, LSE: tl.constexpr,
                  BLOCK_P: tl.constexpr, CHUNK_P: tl.constexpr):
    """Program (column, query head, row): the partials of the partitions that hold the column's keys, merged by their
    maxima and sums (log-sum-exp weights) into the output row (and its log-sum-exp). The sums run over tiles of
    CHUNK_P partitions in partition order, a tile's in one fixed tree: the same order for every launch over a pool
    (BLOCK_P and CHUNK_P follow the partition count), so a column's bits are its own."""
    c = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.program_id(2)
    if COUNTS:
        count = tl.minimum(tl.load(count_ptr + r), COLS)
    else:
        count = count_all
    if c < count:
        start = tl.load(start_ptr + r).to(tl.int64)
        if n_fixed > 0:
            nb = tl.minimum(n_parts, n_fixed)
        else:
            first = tl.load(first_ptr + r).to(tl.int64)
            nb = tl.minimum(n_parts, tl.cdiv(first + c + 1, PIECE)).to(tl.int32)
        base = ((_partial_row(off_ptr, r, COLS, OFFSETS) + c) * HQ + h) * n_parts
        dv = tl.arange(0, DV)
        b = tl.arange(0, BLOCK_P)
        ms = tl.load(pm_ptr + base + b, mask=b < nb, other=float("-inf"))
        top = tl.max(ms, axis=0)
        sub = tl.where(top == float("-inf"), 0.0, top)
        ws = tl.where(ms == float("-inf"), 0.0, tl.exp(ms - sub))
        den = tl.sum(ws * tl.load(pl_ptr + base + b, mask=b < nb, other=0.0), axis=0)
        acc = tl.zeros([DV], tl.float32)
        j = tl.arange(0, CHUNK_P)
        for b0 in range(0, nb, CHUNK_P):
            jb = b0 + j
            mj = tl.load(pm_ptr + base + jb, mask=jb < nb, other=float("-inf"))
            wj = tl.where(mj == float("-inf"), 0.0, tl.exp(mj - sub))
            oj = tl.load(po_ptr + (base + jb)[:, None] * DV + dv[None, :], mask=(jb < nb)[:, None], other=0.0)
            acc = acc + tl.sum(wj[:, None] * oj, axis=0)
        o = tl.where(den > 0, acc / den, 0.0)
        tl.store(out_ptr + (start + c) * stride_ot + h * stride_oh + dv, o.to(out_ptr.dtype.element_ty))
        if LSE:
            tl.store(lse_ptr + (start + c) * HQ + h, tl.where(den > 0, top + tl.log(den), float("-inf")))


# -- prompt chunks ----------------------------------------------------------------------------------------------------
@triton.jit
def _prefill_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, out_ptr, lse_ptr, pos_ptr, fk_ptr, fv_ptr, first_ptr,
                    T, pos0, first0, sm_scale, k_mult, v_mult, stride_qt, stride_qh, stride_ot, stride_oh, stride_fk,
                    stride_fv,
                    HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, PAGE: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr,
                    INT8_QK: tl.constexpr, LSE: tl.constexpr, POS_T: tl.constexpr, FRESH: tl.constexpr,
                    FIRST_T: tl.constexpr):
    """Program (query block, query head): a chunk of one sequence's queries at positions pos0.. over its positions
    [0, own], causal: through its page table, or with FRESH the positions before `first` through it and the chunk's
    own from `first` on from its fresh keys and values (fk_ptr, fv_ptr: [T][HKV][D] in the queries' dtype, as the
    projections gave them; first: first0, or with FIRST_T read from first_ptr on the device, at least pos0); with POS_T
    pos0 is read from pos_ptr on the device."""
    if POS_T:
        pos0 = tl.load(pos_ptr).to(tl.int32)
    if FIRST_T:
        first0 = tl.load(first_ptr).to(tl.int32)
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
    paged = tl.minimum(tl.maximum(first0, pos0), end) if FRESH else end      # the positions read from the pages
    for n0 in range(0, paged, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < paged
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
    if FRESH:
        # The chunk's own positions from `first` on from its fresh 16-bit keys and values, no page scales.
        for n0 in range(paged, end, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            nm = n < end
            t = (n - pos0).to(tl.int64)
            kc = tl.load(fk_ptr + t[:, None] * stride_fk + kvh * D + d[None, :], mask=nm[:, None], other=0.0)
            s = tl.dot(q, tl.trans(kc.to(q.dtype))) * sm_scale
            s = tl.where((n[None, :] <= qpos[:, None]) & nm[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_sub = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_sub[:, None])
            alpha = tl.exp(m_i - m_sub)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            vc = tl.load(fv_ptr + t[:, None] * stride_fv + kvh * D + d[None, :], mask=nm[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), vc.to(q.dtype))
            m_i = m_new
    o = acc / l_i[:, None]
    tl.store(out_ptr + rows[:, None] * stride_ot + head * stride_oh + d[None, :], o.to(out_ptr.dtype.element_ty),
             mask=rmask[:, None])
    if LSE:
        tl.store(lse_ptr + rows * HQ + head, m_i + tl.log(l_i), mask=rmask)
