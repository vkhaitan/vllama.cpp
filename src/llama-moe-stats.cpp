#include "llama-moe-stats.h"
#include "llama-impl.h"

#include <cstdlib>
#include <cstdio>
#include <algorithm>

bool llama_moe_stats_init(llama_moe_stats & stats, uint32_t n_layer, uint32_t n_expert) {
    const char * env = getenv("LLAMA_MOE_EXPERT_STATS");
    stats.enabled = env && atoi(env) > 0;

    if (!stats.enabled) {
        return false;
    }

    if (n_expert == 0) {
        // Not an MoE model
        stats.enabled = false;
        return false;
    }

    stats.n_layer  = n_layer;
    stats.n_expert = n_expert;

    // Initialize counts to zero
    stats.expert_counts.resize(n_layer);
    for (uint32_t l = 0; l < n_layer; ++l) {
        stats.expert_counts[l].assign(n_expert, 0);
    }
    stats.layer_totals.assign(n_layer, 0);

    LLAMA_LOG_INFO("%s: MoE expert statistics tracking enabled (n_layer=%u, n_expert=%u)\n",
                   __func__, n_layer, n_expert);

    return true;
}

void llama_moe_stats_record(
    llama_moe_stats & stats,
    const int32_t   * expert_ids,
    int64_t           n_expert_used,
    int64_t           n_tokens,
    int               il) {

    if (!stats.enabled || il < 0 || (uint32_t)il >= stats.n_layer) {
        return;
    }

    auto & counts = stats.expert_counts[il];
    int64_t total = 0;

    // Iterate over [n_expert_used, n_tokens] tensor
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_expert_used; ++e) {
            int32_t expert_id = expert_ids[e + t * n_expert_used];
            if (expert_id >= 0 && (uint32_t)expert_id < stats.n_expert) {
                counts[expert_id]++;
                total++;
            }
        }
    }

    stats.layer_totals[il] += total;
}

void llama_moe_stats_print(const llama_moe_stats & stats) {
    if (!stats.enabled) {
        return;
    }

    // Check if there's any data to print
    bool has_data = false;
    for (uint32_t l = 0; l < stats.n_layer; ++l) {
        if (stats.layer_totals[l] > 0) {
            has_data = true;
            break;
        }
    }

    if (!has_data) {
        return; // No MoE data collected, skip printing entirely
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "MoE Expert Statistics:\n");
    fprintf(stderr, "======================\n");
    fprintf(stderr, "%5s %7s %10s %7s\n", "Layer", "Expert", "Count", "%");
    fprintf(stderr, "----- ------- ---------- -------\n");

    for (uint32_t l = 0; l < stats.n_layer; ++l) {
        int64_t layer_total = stats.layer_totals[l];
        if (layer_total == 0) {
            continue; // Skip layers with no MoE activity (e.g., dense lead layers)
        }

        // Collect and sort experts by count (descending)
        std::vector<std::pair<uint32_t, int64_t>> sorted_experts;
        for (uint32_t e = 0; e < stats.n_expert; ++e) {
            if (stats.expert_counts[l][e] > 0) {
                sorted_experts.emplace_back(e, stats.expert_counts[l][e]);
            }
        }
        std::sort(sorted_experts.begin(), sorted_experts.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });

        for (const auto & [expert_id, count] : sorted_experts) {
            float pct = 100.0f * count / layer_total;
            fprintf(stderr, "%5u %7u %10lld %6.1f%%\n",
                    l, expert_id, (long long)count, pct);
        }
    }

    fprintf(stderr, "\n");
}

void llama_moe_stats_reset(llama_moe_stats & stats) {
    for (uint32_t l = 0; l < stats.n_layer; ++l) {
        std::fill(stats.expert_counts[l].begin(), stats.expert_counts[l].end(), 0);
    }
    std::fill(stats.layer_totals.begin(), stats.layer_totals.end(), 0);
}
