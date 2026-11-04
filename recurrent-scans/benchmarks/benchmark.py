import torch

from kernels.benchmark import Benchmark


class RecurrentScansBenchmark(Benchmark):
    """Gated-delta prefill of one 2,048-token chunk on Qwen3.8-27B's shape (16 key and 48 value heads, dk = dv = 128)."""

    def setup(self):
        t, kh, vh, d = 2048, 16, 48, 128
        g = torch.Generator(device="cpu").manual_seed(0)
        self.q = torch.nn.functional.normalize(torch.randn(t, kh, d, generator=g), dim=-1).to(self.device)
        self.k = torch.nn.functional.normalize(torch.randn(t, kh, d, generator=g), dim=-1).to(self.device)
        self.v = torch.randn(t, vh, d, generator=g).to(self.device)
        self.log_decay = (-torch.rand(t, vh, generator=g) * 0.1).to(self.device)
        self.beta = torch.rand(t, vh, generator=g).to(self.device)
        self.state0 = (torch.randn(vh, d, d, generator=g) * 0.1).to(self.device)
        self.state = self.state0.clone()
        self.out = torch.empty_like(self.v)

    def benchmark_base(self):
        self.state.copy_(self.state0)
        self.kernel.gated_delta_prefill(self.q, self.k, self.v, self.log_decay, self.beta, self.state, scale=128 ** -0.5,
                                        out=self.out)

    def verify_base(self) -> torch.Tensor:
        # The token-by-token scan of the same recurrence (another summation order, so compared with tolerance).
        state = self.state0.clone()
        return self.kernel.gated_delta_scan(self.q, self.k, self.v, self.log_decay, self.beta, state, scale=128 ** -0.5)
