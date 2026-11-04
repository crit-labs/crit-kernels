#pragma once

#include <optional>

#include <torch/torch.h>

// x [T, channels], weight [channels, width], bias [channels] or none; state [channels, width - 1] in/out (the window
// before row 0 in, the window after the last row out); out [T, channels].
void causal_conv1d(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& weight,
                   const std::optional<torch::Tensor>& bias, bool silu);

// q, k [T, k_heads, dk]; v [T, v_heads, dv]; log_decay, beta [T, v_heads]; state [v_heads, dk, dv] in/out;
// out [T, v_heads, dv]. dk 64 or 128, dv a multiple of 64. Prefill: chunked two-pass. Scan: token by token, column
// invariant, optionally writing the state after every token to checkpoints [T, v_heads, dk, dv].
void gated_delta_prefill(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& q, const torch::Tensor& k,
                         const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta, double scale);
void gated_delta_scan(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& q, const torch::Tensor& k,
                      const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta, double scale,
                      const std::optional<torch::Tensor>& checkpoints);

// x [T, heads, head_dim]; dt [T, heads]; B, C [T, groups, d_state]; A, D [heads]; state [heads, head_dim, d_state]
// in/out; out [T, heads, head_dim]. d_state 64 or 128, head_dim a multiple of 32.
void ssd_prefill(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& dt,
                 const torch::Tensor& B, const torch::Tensor& C, const torch::Tensor& A, const std::optional<torch::Tensor>& D);
void ssd_scan(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& dt,
              const torch::Tensor& B, const torch::Tensor& C, const torch::Tensor& A, const std::optional<torch::Tensor>& D,
              const std::optional<torch::Tensor>& checkpoints);
