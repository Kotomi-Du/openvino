// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "program_helpers.h"
#include "intel_gpu/runtime/itt.hpp"
#include <vector>

using namespace cldnn;

void basic_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::BasicMemoryDependencies");
    auto itr = p.get_processing_order().begin();
    std::vector<size_t> past_outputs;

    // Pre-compute each node's position in the processing order and the
    // position of its last consumer.  Used below to detect long-lived
    // buffers that span many nodes in the schedule.
    std::unordered_map<size_t, size_t> node_pos;       // unique_id → ordinal
    std::unordered_map<size_t, size_t> last_consumer_pos;  // unique_id → max-user ordinal
    {
        size_t ordinal = 0;
        for (auto it = p.get_processing_order().begin(); it != p.get_processing_order().end(); ++it) node_pos[(*it)->get_unique_id()] = ordinal++;
        for (auto it = p.get_processing_order().begin(); it != p.get_processing_order().end(); ++it) {
            size_t lcp = 0;
            for (auto* u : (*it)->get_users()) {
                auto f = node_pos.find(u->get_unique_id());
                if (f != node_pos.end() && f->second > lcp) lcp = f->second;
            }
            last_consumer_pos[(*it)->get_unique_id()] = lcp;
        }
    }       
    
    // Buffers whose output is still live (written but last consumer not yet reached).
    std::unordered_map<size_t, size_t> live_buffers;  // unique_id → last-consumer ordinal
    

    
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr;
        itr++;

        // data primitive can't be reused
        if (node->is_type<data>())
            continue;

        const size_t my_pos = node_pos[node->get_unique_id()];

        // Expire dead buffers, then collect every surviving live buffer id.
        // Adding them as memory-dependencies prevents the pool from reusing
        // a buffer that was written at an earlier processing position but
        // whose data is still needed by a later consumer.
        std::vector<size_t> live_ids;
        for (auto it = live_buffers.begin(); it != live_buffers.end();) {
            if (my_pos > it->second) {
                it = live_buffers.erase(it);
            
            }
            else {
                live_ids.push_back(it->first);
                ++it;
                
            }
            
        }
        if (node->can_share_buffer() && !live_ids.empty()) 
            node->add_memory_dependency(live_ids);
       
        


        // add my dependencies to restriction list (can't share input.output buffers)
        for (const auto& it : node->get_dependencies()) {
            add_memory_dependency(node, it.first);
            add_memory_dependency(it.first, node);
        }

        // LoRA can reuse the memory of the previous node, but not be optimized
        // So memory dependencies need to be expanded with LoRA dependencies
        if (node->have_user_with_type<lora>()) {
            for (const auto& it : node->get_users().front()->get_dependencies()) {
                add_memory_dependency(node, it.first);
                add_memory_dependency(it.first, node);
            }
        }

        if (node->get_preferred_impl_type() == impl_types::onednn) {
            size_t eltw_dep = 0;
            for (auto& fused_op : node->get_fused_primitives()) {
                if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1) {
                    // If it is first sum, reuse the buffer
                    auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(*node, fused_op);
                    if (fusing_type != add_fusing_type::sum || eltw_dep != 0)
                        continue;
                    if (!fused_op.has_outer_dep())
                        continue;
                    eltw_dep = fused_op.outer_dep_start_idx;
                    auto& eltw_node = node->get_dependency(eltw_dep);
                    node->can_share_buffer(false);
                    // Walk up the in-place chain to mark the root buffer owner as non-shareable.
                    auto* root = &eltw_node;
                    while (root->can_be_optimized() && !root->get_dependencies().empty()) {
                        root = &root->get_dependency(0);
                    }
                    root->can_share_buffer(false);

                    for (auto& user : node->get_users()) {
                        add_memory_dependency(user, &eltw_node);
                        add_memory_dependency(user, node);
                    }
                }
            }
        }

        // Register long-lived buffers: if this node's output is consumed
        // much later (> 64 nodes gap), track it so that every intermediate
        // node acquires a memory-dependency on it.  Short-lived buffers are
        // already protected by the direct-dependency logic above.
        if (node->may_use_mempool()) {
            const size_t lcp = last_consumer_pos[node->get_unique_id()];
            if (lcp > my_pos + 64) live_buffers[node->get_unique_id()] = lcp;
            
        }
        

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs,
        // this output has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output()) {
            past_outputs.push_back(node->get_unique_id());
            if (node->is_type<mutable_data>()) {
                // if output is mutable data, then propagate output flag to its dependencies
                for (auto& dep : node->get_dependencies()) {
                    dep.first->set_output(true);
                }
            }
        }
    }
}
