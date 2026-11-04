// Grouped expert projections (expert-streamed-prefill) on CUDA: a whole routed MoE block (route sort, gather, gate/up,
// SwiGLU, down, combine) in each weight format against an FP64 host reference of the same decoded weights; the same
// bits on a repeat and for a chunk's first tokens served as a chunk of their own (a token's result does not depend
// on its chunk-mates); experts read from device memory and from mapped host memory through the pointer table; a
// skipped route, an expert without routes, and rejected launches.
#include "grouped_experts.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gx = crit::techniques::streamed_prefill;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
template<class T> T* dev(std::size_t n) { void* p = nullptr; ok(cudaMalloc(&p, (n ? n : 1) * sizeof(T)), "malloc"); return static_cast<T*>(p); }
template<class T> std::vector<T> get(const T* p, std::size_t n) {
    std::vector<T> v(n);
    ok(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
template<class T> void put(T* p, const std::vector<T>& v) { ok(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }

double e4m3(std::uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e == 0 ? std::ldexp(m, -9) : std::ldexp(8 + m, e - 10);
    return b & 0x80 ? -v : v;
}
double e2m1(unsigned code) {
    const double magnitude[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return code & 8u ? -magnitude[code & 7u] : magnitude[code & 7u];
}
double half_value(__half h) { return static_cast<double>(__half2float(h)); }
double round_half(double v) { return half_value(__float2half_rn(static_cast<float>(v))); }

// One expert matrix [rows][cols] in `format`: its stored bytes and its decoded values (what the kernel multiplies).
struct Stored {
    std::vector<std::uint8_t> codes, scales;
    float post = 1.f;
    std::vector<double> values;  // [rows][cols], post and row scales included
};

Stored make(gx::Format f, int rows, int cols, std::mt19937& rng) {
    Stored s;
    s.values.assign(std::size_t(rows) * cols, 0.0);
    std::uniform_int_distribution<int> byte(0, 255), nibble(0, 15);
    auto moderate_e4m3 = [&](int lo, int hi) {  // positive, exponent field in [lo, hi]
        return static_cast<std::uint8_t>(((lo + byte(rng) % (hi - lo + 1)) << 3) | (byte(rng) & 7));
    };
    if(f == gx::Format::bf16) {
        std::normal_distribution<float> normal(0.f, 0.05f);
        s.codes.resize(std::size_t(rows) * cols * 2);
        for(std::size_t i = 0; i < s.values.size(); ++i) {
            const float w = normal(rng);
            std::uint32_t bits;
            std::memcpy(&bits, &w, 4);
            const std::uint16_t top = static_cast<std::uint16_t>(bits >> 16);  // truncation is a valid bf16
            std::memcpy(&s.codes[2 * i], &top, 2);
            const std::uint32_t back = std::uint32_t(top) << 16;
            float v;
            std::memcpy(&v, &back, 4);
            s.values[i] = round_half(v);
        }
    } else if(f == gx::Format::fp8_e4m3_row) {
        s.codes.resize(std::size_t(rows) * cols);
        s.scales.resize(std::size_t(rows) * 4);
        for(int n = 0; n < rows; ++n) {
            const float row_scale = 0.01f * (1 + n % 5);
            std::memcpy(&s.scales[std::size_t(n) * 4], &row_scale, 4);
            for(int k = 0; k < cols; ++k) {
                const std::uint8_t c = static_cast<std::uint8_t>(moderate_e4m3(3, 8) | (byte(rng) & 0x80));
                s.codes[std::size_t(n) * cols + k] = c;
                s.values[std::size_t(n) * cols + k] = e4m3(c) * row_scale;
            }
        }
    } else {
        const bool nv = f == gx::Format::nvfp4;
        const int block = nv ? 16 : 32;
        s.post = nv ? 0.05f : 1.f;
        s.codes.resize(std::size_t(rows) * cols / 2);
        s.scales.resize(std::size_t(rows) * cols / block);
        for(auto& b : s.scales) b = nv ? moderate_e4m3(5, 8) : static_cast<std::uint8_t>(127 - 6 + byte(rng) % 4);
        for(int n = 0; n < rows; ++n)
            for(int k = 0; k < cols; k += 2) {
                const unsigned lo = unsigned(nibble(rng)), hi = unsigned(nibble(rng));
                s.codes[(std::size_t(n) * cols + k) / 2] = static_cast<std::uint8_t>(lo | (hi << 4));
                const std::uint8_t sb = s.scales[(std::size_t(n) * cols + k) / block];
                const double scale = nv ? e4m3(sb) : std::ldexp(1.0, int(sb) - 127);
                s.values[std::size_t(n) * cols + k] = e2m1(lo) * scale * s.post;
                s.values[std::size_t(n) * cols + k + 1] = e2m1(hi) * scale * s.post;
            }
    }
    return s;
}

// Device (or mapped host) copies of an expert's bytes; returns the Matrix the kernel reads.
gx::Matrix upload(const Stored& s, bool mapped, std::vector<void*>& hosts) {
    auto place = [&](const std::vector<std::uint8_t>& bytes) -> const std::uint8_t* {
        if(bytes.empty()) return nullptr;
        if(!mapped) {
            auto* d = dev<std::uint8_t>(bytes.size());
            put(d, bytes);
            return d;
        }
        void* h = nullptr;
        ok(cudaHostAlloc(&h, bytes.size(), cudaHostAllocMapped), "mapped expert");
        std::memcpy(h, bytes.data(), bytes.size());
        hosts.push_back(h);
        void* d = nullptr;
        ok(cudaHostGetDevicePointer(&d, h, 0), "mapped pointer");
        return static_cast<const std::uint8_t*>(d);
    };
    gx::Matrix m;
    m.codes = place(s.codes);
    m.scales = place(s.scales);
    m.post = s.post;
    return m;
}

struct Block {
    int tokens = 0, top_k = 0, experts = 0, hidden = 0, inter = 0;
    gx::Matrix* gate_up = nullptr;  // device tables [experts]
    gx::Matrix* down = nullptr;
};

// Runs the routed block on the first `tokens` tokens of x / ids / weights; returns out [tokens][hidden].
std::vector<float> forward(const Block& b, gx::Format f, const __half* x, const std::int32_t* ids, const float* w, int tokens) {
    const int rows = tokens * b.top_k;
    gx::Routing r{ids, w, tokens, b.top_k, b.experts};
    gx::Workspace ws{dev<int>(b.experts), dev<int>(b.experts + 1), dev<int>(b.experts + 1), dev<int>(rows), dev<int>(rows)};
    auto* xp = dev<__half>(std::size_t(rows) * b.hidden);
    auto* gu = dev<__half>(std::size_t(rows) * 2 * b.inter);
    auto* act = dev<__half>(std::size_t(rows) * b.inter);
    auto* yp = dev<__half>(std::size_t(rows) * b.hidden);
    auto* out = dev<float>(std::size_t(tokens) * b.hidden);
    ok(gx::enqueue_route_sort(r, ws, nullptr), "route sort");
    ok(gx::enqueue_gather(x, b.hidden, r, ws, xp, nullptr), "gather");
    ok(gx::enqueue_grouped_projection(f, b.gate_up, xp, b.hidden, 2 * b.inter, r, ws, gu, nullptr), "gate/up");
    ok(gx::enqueue_swiglu(gu, b.inter, rows, act, nullptr), "swiglu");
    ok(gx::enqueue_grouped_projection(f, b.down, act, b.inter, b.hidden, r, ws, yp, nullptr), "down");
    ok(gx::enqueue_combine(yp, b.hidden, r, ws, out, nullptr), "combine");
    ok(cudaDeviceSynchronize(), "sync");
    auto result = get(out, std::size_t(tokens) * b.hidden);
    for(void* p : {static_cast<void*>(ws.counts), static_cast<void*>(ws.offsets), static_cast<void*>(ws.tile_offsets),
                   static_cast<void*>(ws.row_route), static_cast<void*>(ws.route_row), static_cast<void*>(xp),
                   static_cast<void*>(gu), static_cast<void*>(act), static_cast<void*>(yp), static_cast<void*>(out)})
        cudaFree(p);
    return result;
}

void run(gx::Format f, const char* name) {
    constexpr int T = 150, K = 4, E = 12, H = 128, I = 64;
    std::mt19937 rng(3 + int(f));
    std::vector<Stored> gate_up, down;
    std::vector<gx::Matrix> gate_up_table, down_table;
    std::vector<void*> hosts;
    for(int e = 0; e < E; ++e) {
        gate_up.push_back(make(f, 2 * I, H, rng));
        down.push_back(make(f, H, I, rng));
        gate_up_table.push_back(upload(gate_up.back(), e % 3 == 0, hosts));  // every third expert in mapped host memory
        down_table.push_back(upload(down.back(), e % 3 == 0, hosts));
    }
    Block b{T, K, E, H, I, dev<gx::Matrix>(E), dev<gx::Matrix>(E)};
    put(b.gate_up, gate_up_table);
    put(b.down, down_table);

    std::vector<__half> x(std::size_t(T) * H);
    std::uniform_real_distribution<float> unit(-1.f, 1.f);
    for(auto& v : x) v = __float2half_rn(unit(rng));
    std::vector<std::int32_t> ids(std::size_t(T) * K);
    std::vector<float> weights(ids.size());
    for(int t = 0; t < T; ++t)
        for(int j = 0; j < K; ++j) {
            ids[std::size_t(t) * K + j] = int((t * 7 + j * 3 + (t % 5 == 0 ? j : 0)) % (E - 1));  // expert E - 1 unused
            weights[std::size_t(t) * K + j] = 0.1f + 0.2f * j;
        }
    ids[std::size_t(17) * K + 2] = -1;  // a skipped route
    auto* dx = dev<__half>(x.size());
    auto* dids = dev<std::int32_t>(ids.size());
    auto* dw = dev<float>(weights.size());
    put(dx, x);
    put(dids, ids);
    put(dw, weights);

    const auto full = forward(b, f, dx, dids, dw, T);
    const auto again = forward(b, f, dx, dids, dw, T);
    require(std::memcmp(full.data(), again.data(), full.size() * sizeof(float)) == 0, std::string(name) + ": same bits on a repeat");
    const int part = 70;
    const auto first = forward(b, f, dx, dids, dw, part);
    require(std::memcmp(full.data(), first.data(), first.size() * sizeof(float)) == 0,
            std::string(name) + ": a token's result depends on its chunk-mates");

    double worst = 0, scale = 0;
    for(int t = 0; t < T; ++t)
        for(int h = 0; h < H; ++h) {
            double acc = 0;
            for(int j = 0; j < K; ++j) {
                const int e = ids[std::size_t(t) * K + j];
                if(e < 0) continue;
                const auto& gu = gate_up[e].values;
                const auto& dn = down[e].values;
                double y = 0;
                for(int i = 0; i < I; ++i) {
                    double g = 0, u = 0;
                    for(int k = 0; k < H; ++k) {
                        const double xv = half_value(x[std::size_t(t) * H + k]);
                        g += gu[std::size_t(i) * H + k] * xv;
                        u += gu[std::size_t(I + i) * H + k] * xv;
                    }
                    y += dn[std::size_t(h) * I + i] * (g / (1 + std::exp(-g)) * u);
                }
                acc += weights[std::size_t(t) * K + j] * y;
            }
            worst = std::max(worst, std::abs(acc - full[std::size_t(t) * H + h]));
            scale = std::max(scale, std::abs(acc));
        }
    require(scale > 0 && worst <= 2e-2 * scale, std::string(name) + ": differs from the FP64 reference");
    std::cout << name << ": " << T << " tokens, max |error| " << worst << " of max |value| " << scale
              << "; same bits on a repeat and for the first " << part << " tokens alone\n";
    for(void* h : hosts) cudaFreeHost(h);
}
} // namespace

int main() {
    try {
        run(gx::Format::bf16, "bf16");
        run(gx::Format::fp8_e4m3_row, "fp8_e4m3_row");
        run(gx::Format::nvfp4, "nvfp4");
        run(gx::Format::mxfp4, "mxfp4");
        gx::Routing r{nullptr, nullptr, 1, 1, 1};
        gx::Workspace w{};
        require(gx::enqueue_route_sort(r, w, nullptr) == cudaErrorInvalidValue, "null routing rejected");
        std::int32_t* ids = dev<std::int32_t>(1);
        r.ids = ids;
        int* scratch = dev<int>(4);
        w = gx::Workspace{scratch, scratch, scratch, scratch, scratch};
        gx::Matrix* table = dev<gx::Matrix>(1);
        __half* buffer = dev<__half>(64 * 64);
        require(gx::enqueue_grouped_projection(gx::Format::nvfp4, table, buffer, 48, 64, r, w, buffer, nullptr) ==
                    cudaErrorInvalidValue, "input width not a multiple of 32 rejected");
        require(gx::enqueue_grouped_projection(gx::Format::nvfp4, table, buffer, 64, 96, r, w, buffer, nullptr) ==
                    cudaErrorInvalidValue, "output width not a multiple of 64 rejected");
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
