#pragma once

#include "llama-hparams.h"

#include <cstdint>
#include <vector>

// MoE Expert Statistics
// Track which experts are used during inference (controlled by LLAMA_MOE_EXPERT_STATS env var)

struct llama_moe_stats {
    bool enabled = false;
    uint32_t n_layer  = 0;
    uint32_t n_expert = 0;

    // expert_counts[layer][expert] = count of times expert was selected
    std::vector<std::vector<int64_t>> expert_counts;

    // total selections per layer (for percentage calculation)
    std::vector<int64_t> layer_totals;
};

// Initialize MoE stats tracking (checks LLAMA_MOE_EXPERT_STATS env var)
// Returns true if stats tracking is enabled
bool llama_moe_stats_init(llama_moe_stats & stats, uint32_t n_layer, uint32_t n_expert);

// Record expert usage from selected_experts tensor data
// expert_ids: pointer to I32 tensor data [n_expert_used, n_tokens]
// n_expert_used: number of experts selected per token
// n_tokens: number of tokens in the batch
// il: layer index
void llama_moe_stats_record(
    llama_moe_stats & stats,
    const int32_t   * expert_ids,
    int64_t           n_expert_used,
    int64_t           n_tokens,
    int               il);

// Print summary table to stderr
void llama_moe_stats_print(const llama_moe_stats & stats);

// Reset all accumulated statistics
void llama_moe_stats_reset(llama_moe_stats & stats);
