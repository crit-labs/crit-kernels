import torch

from kernels.benchmark import Benchmark


class FusedNormsBenchmark(Benchmark):
    """The norm after a block's output, at a 5,120-wide hidden row and 8 decode rows: the block's BF16 output added to
    an FP32 residual stream in place, then RMSNorm with a zero-centred weight to BF16. benchmarks/shapes.py measures
    every function at three models' shapes against the torch chains."""

    def setup(self):
        g = torch.Generator(device="cpu").manual_seed(0)
        self.x = torch.randn(8, 5120, generator=g).to(self.device, torch.bfloat16)
        self.residual0 = torch.randn(8, 5120, generator=g).to(self.device)
        self.residual = self.residual0.clone()
        self.weight = (torch.randn(5120, generator=g) * 0.1).to(self.device)

    def benchmark_base(self):
        self.residual.copy_(self.residual0)
        self.out = self.kernel.rms_norm(self.x, self.weight, 1e-6, offset=1.0, residual=self.residual,
                                        out_dtype=torch.bfloat16)[0]

    def verify_base(self) -> torch.Tensor:
        # The torch chain a model writes by hand, the same function in another summation order.
        s = self.residual0 + self.x.float()
        return (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + self.weight)).to(torch.bfloat16)
