#pragma once
// The reference scans of recurrent-state-recovery's device commit strategies, moved here with their fixture: a verify
// writing per-column checkpoints (Strategy A) or keeping one running copy (Strategy B), the recompute of an accepted
// prefix from the round-start state (Strategy B's commit) and the deferred verify (Strategy C: replay the recorded prefix
// from a compact stash, then verify writing the stash). One block per head, one thread per value column; FP32; not for
// performance (the pool rows of recurrent_rows.h are). The commit kernels these pair with (select_accepted,
// record_accepted) and DeviceGeometry are recurrent-state-recovery's (crit, code/recurrent_cuda.h).
#include "recurrent_cuda.h"

namespace crit::techniques::recurrent::cuda {
// Projected, gated step inputs for `columns` positions; entry (t, h) is at
// query/key[(t * heads + h) * keys], value[(t * heads + h) * values],
// log_decay/beta[t * heads + h]. log_decay <= 0, beta in [0, 1].
struct DeviceSteps {
    const float* query = nullptr;
    const float* key = nullptr;
    const float* value = nullptr;
    const float* log_decay = nullptr;
    const float* beta = nullptr;
};
// Verify `columns` steps from `start` and write outputs[(t * heads + h) * values + v].
// Strategy A (checkpoints): non-null `checkpoints` receives the state after every
// column, [columns][state_elements]. Strategy B (recompute): null `checkpoints`
// and a `work` buffer of state_elements; only outputs survive.
cudaError_t enqueue_verify(const DeviceGeometry&, const float* start, DeviceSteps, std::uint32_t columns,
                           float* outputs, float* checkpoints, float* work, cudaStream_t);
// Strategy B commit: state = start advanced through the first `accepted` saved
// steps (accepted read from device memory, clamped to `columns`). state != start.
cudaError_t enqueue_recompute_accepted(const DeviceGeometry&, const float* start, DeviceSteps,
                                       const std::int32_t* accepted, std::uint32_t columns, float* state, cudaStream_t);

// Strategy C (deferred): one live state, no round-start copy. Verify first
// replays the previous round's pending accepted prefix from a compact stash into
// the live state, then verifies the new columns and overwrites the stash. Commit
// only records the accepted count, so it is a scalar write that can rerun with a
// smaller count after a host truncation. Stash layout for `capacity` columns:
// key[(t * heads + h) * keys] (normalized key), decay[t * heads + h] (exp(log_decay)),
// delta[(t * heads + h) * values] (beta * residual).
struct DeferredStash {
    float* key = nullptr;
    float* decay = nullptr;
    float* delta = nullptr;
    std::uint32_t capacity = 0; // stash columns; columns <= capacity
};
// Applies min(*pending, capacity) stashed steps to `state`, then verifies
// `columns` steps (columns may be zero: apply only) and zeroes *pending, so each
// recorded prefix is applied once. `work` holds one running copy.
cudaError_t enqueue_verify_deferred(const DeviceGeometry&, float* state, DeviceSteps, std::uint32_t columns,
                                    std::int32_t* pending, DeferredStash, float* outputs, float* work, cudaStream_t);
}
