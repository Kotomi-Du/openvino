// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// 1. Trivial case (greedy search, no state initializer)
///     ┌───────────┐      ┌───────────┐                                      ┌───────────┐
///     │ ReadValue │      │  SomeOp   │                                      │  SomeOp   │
///     | (past_kv) |      | new_token |                                      | new_token |
///     └─────┬─────┘      └─────┬─────┘                                      └─────┬─────┘
///           │                  │                                                  |
///           │                  |                                                  |
///           │   ┌────────┐     │                                             ┌────┴────┐       ┌──────────┐
///           └───┤ Concat ├─────┘                    =>                       | KVCache |.......| Variable |
///               └───┬────┘                                                   └────┬────┘       └──────────┘
///                   │                                                             |
///        ┌──────────┴────────────┐                                                |
///   ┌────┴────────┐         ┌────┴──────┐                                    ┌────┴────┐
///   │  Assign     │         │  SomeOp   │                                    | SomeOp  |
///   | (present_kv |         |  (SDPA)   |                                    | (SDPA)  |
///   └─────────────┘         └───────────┘                                    └─────────┘

/// 2. With gather for beam search (or model which supports both greedy and beam search)
///     ┌───────────┐      ┌────────────┐    ┌───────────┐                   ┌───────────┐      ┌────────────┐    ┌───────────┐
///     │ ReadValue │      │ Parameter  │    │  SomeOp   │                   │ ReadValue │      │ Parameter  │    │  SomeOp   │
///     | (past_kv) |      |  beam_idx  |    | new_token |                   | (past_kv) |      |  beam_idx  |    | new_token |
///     └─────┬─────┘      └─────┬──────┘    └─────┬─────┘                   └─────┬─────┘      └─────┬──────┘    └─────┬─────┘
///           │                  │                 │                               │                  │                 │
///     ┌─────┴──────┐           |                 │                         ┌─────┴──────┐           |                 │
///     |   Gather   |───────────┘                 │                         |   Gather   |───────────┘                 │
///     └─────┬──────┘                             │                         └─────┬──────┘                             │
///           |                                    │                               |                        ┌───────────┘
///           |                                    │                               |                        |
///           │                                    │                               |                        |
///           │   ┌────────┐                       │                               |            ┌───────────┴───────┐                       ┌──────────┐
///           └───┤ Concat ├───────────────────────┘        =>                     └────────────┤      KVCache      |.......................| Variable |
///               └───┬────┘                                                                    └────┬──────────────┘                       └──────────┘
///                   │                                                                              |
///        ┌──────────┴─────────────┐                                                                |
///   ┌────┴─────────┐         ┌────┴──────┐                                                    ┌────┴────┐
///   │  Assign      │         │  SomeOp   │                                                    | SomeOp  |
///   | (present_kv) |         |  (SDPA)   |                                                    | (SDPA)  |
///   └──────────────┘         └───────────┘                                                    └─────────┘

/// 3. Similar to case 2, but with variable initializer
///     ┌─────────────────────┐                                               ┌─────────────────────┐
///     │       SomeOp        │                                               │       SomeOp        │
///     | (state initializer) |                                               | (state initializer) |
///     └─────┬───────────────┘                                               └─────┬───────────────┘
///           |                                                                     |
///     ┌─────┴─────┐      ┌────────────┐    ┌───────────┐                    ┌─────┴─────┐      ┌────────────┐    ┌───────────┐
///     │ ReadValue │      │ Parameter  │    │  SomeOp   │                    │ ReadValue │      │ Parameter  │    │  SomeOp   │
///     | (past_kv) |      |  beam_idx  |    | new_token |                    | (past_kv) |      |  beam_idx  |    | new_token |
///     └─────┬─────┘      └─────┬──────┘    └─────┬─────┘                    └─────┬─────┘      └─────┬──────┘    └─────┬─────┘
///           │                  │                 │                                │                  │                 │
///     ┌─────┴──────┐           |                 │                          ┌─────┴──────┐           |                 │
///     |   Gather   |───────────┘                 │                          |   Gather   |───────────┘                 │
///     └─────┬──────┘                             │                          └─────┬──────┘                             │
///           |                                    │                                |                        ┌───────────┘
///           |                                    │                                |                        |
///           │                                    │                                |                        |
///           │   ┌────────┐                       │                                |            ┌───────────┴───────┐                       ┌──────────┐
///           └───┤ Concat ├───────────────────────┘        =>                      └────────────┤      KVCache      |.......................| Variable |
///               └───┬────┘                                                                     └────┬──────────────┘                       └──────────┘
///                   │                                                                               |
///        ┌──────────┴────────────┐                                                                  |
///   ┌────┴────────┐         ┌────┴──────┐                                                      ┌────┴────┐
///   │  Assign     │         │  SomeOp   │                                                      | SomeOp  |
///   | (present_kv |         |  (SDPA)   |                                                      | (SDPA)  |
///   └─────────────┘         └───────────┘                                                      └─────────┘
class KVCacheFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KVCacheFusionMatcher");
    KVCacheFusionMatcher();
};

class KVCacheFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("KVCacheFusion");
    KVCacheFusion();

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class StatelessKVFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("StatelessKVFusionMatcher");
    StatelessKVFusionMatcher();

private:
    struct CachedNodes {
        std::shared_ptr<ov::Node> total_seqlen;
        std::shared_ptr<ov::Node> seqlens_k;
        ov::Output<ov::Node> present_kv_len;

        struct TrimmedMask {
            int64_t new_token_len = 0;
            ov::Output<ov::Node> original_mask;
            std::shared_ptr<ov::Node> trimmed_mask;
        };
        std::vector<TrimmedMask> trimmed_masks;
    };
    std::map<std::shared_ptr<ov::Node>, std::shared_ptr<CachedNodes>> m_cache;
    std::set<ov::Output<ov::Node>> m_trimmed_masks;
};

class StatelessKVFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("StatelessKVFusion");
    StatelessKVFusion();

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};


}   // namespace ov::intel_gpu
