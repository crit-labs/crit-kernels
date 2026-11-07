import math

import torch

from kernels.benchmark import Benchmark


class PagedKvAttentionBenchmark(Benchmark):
    """A verify window of 5 columns over 32,768 positions of FP8 pages with a scale per position and KV head, at a
    24 query / 4 KV head x 256 shape, 64-position pages."""

    def setup(self):
        hq, hkv, d, page, length, cols = 24, 4, 256, 64, 32768, 5
        g = torch.Generator(device="cpu").manual_seed(0)
        pages = math.ceil((length + cols) / page)
        x = torch.randn(pages * page, hkv, d, generator=g).to(self.device)
        self.k = torch.zeros(pages, page, hkv, d, dtype=torch.float8_e4m3fn, device=self.device)
        self.v = torch.zeros_like(self.k)
        self.ks = torch.zeros(pages, page, hkv, device=self.device)
        self.vs = torch.zeros_like(self.ks)
        self.table = torch.randperm(pages, generator=g).to(torch.int32).to(self.device)[None]
        slots = self.table[0].long()[:, None] * page + torch.arange(page, device=self.device)[None, :]
        self.kernel.write(x, x, slots.reshape(-1), self.k, self.v, "fp8:row", self.ks, self.vs)
        self.q = torch.randn(cols, hq, d, generator=g).to(self.device).to(torch.bfloat16)
        self.rows = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.firsts = torch.tensor([length], dtype=torch.int64, device=self.device)
        self.out = torch.empty(cols, hq, d, dtype=torch.bfloat16, device=self.device)
        self.cols, self.page = cols, page

    def benchmark_base(self):
        self.kernel.attend(self.q, self.k, self.v, "fp8:row", self.table, self.rows, self.rows, self.firsts, self.cols,
                           self.page, key_scales=self.ks, value_scales=self.vs, out=self.out)

    def verify_base(self) -> torch.Tensor:
        # Each column decoded alone: the same bits as inside the window.
        found = []
        for c in range(self.cols):
            found.append(self.kernel.attend(self.q[c:c + 1], self.k, self.v, "fp8:row", self.table, self.rows,
                                            self.rows, self.firsts + c, 1, self.page, key_scales=self.ks,
                                            value_scales=self.vs))
        return torch.cat(found)
