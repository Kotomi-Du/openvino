// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/runtime/debug_configuration.hpp>

#include "openvino/util/file_util.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>
#include "scaled_dot_product_attention_inst.h"

#include <cstddef>
#include <vector>

using namespace cldnn;
using namespace ::tests;
// #define ENABLE_ONEDNN_FOR_GPU
namespace  {
#ifdef ENABLE_ONEDNN_FOR_GPU
struct sdpa_test_params {
    int head_size;
    int num_heads;
    int sequence_length_q;
    int sequence_length_kv;
    int batch;
    bool dynamic;
    bool use_scalar_scale_val;
    float scale_val;
    bool use_scalar_attn_mask;
    float attn_mask_val;
    bool is_causal;

    // Constructor for basic tests (backward compatibility)
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b, bool dynamic_shape, bool causal = false)
        : head_size(h_size), num_heads(n_heads), sequence_length_q(seq_q),
          sequence_length_kv(seq_kv), batch(b), dynamic(dynamic_shape),
          use_scalar_scale_val(false), scale_val(1.0f), use_scalar_attn_mask(false),
          attn_mask_val(0.0f), is_causal(causal) {}

    // Constructor for advanced caching tests
    sdpa_test_params(int h_size, int n_heads, int seq_q, int seq_kv, int b,
                     bool use_scale, float scale, bool use_mask, float mask, bool causal = false)
        : head_size(h_size), num_heads(n_heads), sequence_length_q(seq_q), sequence_length_kv(seq_kv),
          batch(b), dynamic(true), use_scalar_scale_val(use_scale),
          scale_val(scale), use_scalar_attn_mask(use_mask), attn_mask_val(mask), is_causal(causal) {}
};

struct sdpa_gpu_test : public ::testing::TestWithParam<sdpa_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void load_input(cldnn::memory::ptr mem, size_t idx) {
        auto shapes = mem->get_layout().get_shape();
        size_t size = ov::shape_size(shapes);
        auto input_data = rg.generate_random_1d<ov::float16>(size, -1.0f, 1.0f);
        set_values(mem, input_data);
    }

    std::tuple<cldnn::memory::ptr, cldnn::network::ptr> run_network(bool is_caching_test, bool use_optimized_sdpa,
            cldnn::layout input0_layout,
            cldnn::layout input1_layout,
            cldnn::layout input2_layout,
            cldnn::layout input3_layout,
            cldnn::memory::ptr input0,
            cldnn::memory::ptr input1,
            cldnn::memory::ptr input2,
            cldnn::memory::ptr input3,
            bool use_scalar_scale_val = false,
            float scale_val = 1.0f,
            bool use_scalar_attn_mask = false,
            float attn_mask_val = 0.0f,
            bool is_causal = false) {
        auto& engine = get_test_engine();
        topology topo;
        topo.add(input_layout("input0", input0_layout));
        topo.add(input_layout("input1", input1_layout));
        topo.add(input_layout("input2", input2_layout));
        topo.add(input_layout("input3", input3_layout));

        auto sdpa_prim = scaled_dot_product_attention("sdpa", {input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3")},
            is_causal, -1, {0,2,1,3}, {0,2,1,3}, {0,2,1,3}, {0,1,2,3}, {}, false);

        if (use_scalar_scale_val) {
            sdpa_prim.scale_val = scale_val;
        }

        if (use_scalar_attn_mask) {
            sdpa_prim.attn_mask_val = attn_mask_val;
        }

        topo.add(sdpa_prim);
        topo.add(reorder("result",input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        if (use_optimized_sdpa) {
            if (!is_caching_test) {
                if (engine.get_device_info().supports_immad) {
                    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                        {"sdpa", {format::type::bfyx, "sdpa_micro"}} }));
                } else {
                    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                        {"sdpa", {format::type::bfyx, "sdpa_opt"}} }));
                }
            }
        } else {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
                   {"sdpa", {format::type::bfyx, "sdpa_ref"}} }));
            config.set_user_property(ov::intel_gpu::use_onednn(false));
        }

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("input0", input0);
        net->set_input_data("input1", input1);
        net->set_input_data("input2", input2);
        net->set_input_data("input3", input3);

        auto outputs = net->execute();
        auto output = outputs.at("result").get_memory();
        return {output, net};
    }

    void execute(sdpa_test_params& p, bool is_caching_test = false) {
        const auto head_size = p.head_size;
        const auto num_heads = p.num_heads;
        const auto seq_length_q = p.sequence_length_q;
        const auto seq_length_kv = p.sequence_length_kv;
        const auto batch = p.batch;
        const auto use_scalar_scale_val = p.use_scalar_scale_val;
        const auto scale_val = p.scale_val;
        const auto use_scalar_attn_mask = p.use_scalar_attn_mask;
        const auto attn_mask_val = p.attn_mask_val;
        const auto is_causal = p.is_causal;
        const auto test_two_rank_mask = p.sequence_length_q == p.sequence_length_kv ? true : false;

        auto& engine = get_test_engine();
        cldnn::layout input0_layout, input1_layout, input2_layout, input3_layout;
        cldnn::layout input0_static_layout, input1_static_layout, input2_static_layout, input3_static_layout;

        if (p.dynamic) {
            input0_layout = cldnn::layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
            input1_layout = cldnn::layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
            input2_layout = cldnn::layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);

            if (test_two_rank_mask) {
                input3_layout = cldnn::layout({-1, -1}, data_types::f16, format::bfyx);
            } else {
                input3_layout = cldnn::layout({-1, num_heads, -1, -1}, data_types::f16, format::bfyx);
            }

            input0_static_layout = cldnn::layout({batch, seq_length_q, num_heads, head_size}, data_types::f16, format::bfyx);
            input1_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
            input2_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
            if (test_two_rank_mask) {
                input3_static_layout = cldnn::layout({seq_length_q, seq_length_kv}, data_types::f32, format::bfyx);
            } else {
                input3_static_layout = cldnn::layout({batch, num_heads, 1, seq_length_kv}, data_types::f16, format::bfyx);
            }
        } else {
            input0_static_layout = cldnn::layout({batch, seq_length_q, num_heads, head_size}, data_types::f16, format::bfyx);
            input1_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
            input2_static_layout = cldnn::layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);

            if (test_two_rank_mask) {
                input3_static_layout = cldnn::layout({seq_length_q, seq_length_kv}, data_types::f16, format::bfyx);
            } else {
                input3_static_layout = cldnn::layout({batch, num_heads, 1, seq_length_kv}, data_types::f16, format::bfyx);
            }

            input0_layout = input0_static_layout;
            input1_layout = input1_static_layout;
            input2_layout = input2_static_layout;
            input3_layout = input3_static_layout;
        }

        auto input0 = engine.allocate_memory(input0_static_layout);
        auto input1 = engine.allocate_memory(input1_static_layout);
        auto input2 = engine.allocate_memory(input2_static_layout);
        auto input3 = engine.allocate_memory(input3_static_layout);

        load_input(input0, 0);
        load_input(input1, 1);
        load_input(input2, 2);
        load_input(input3, 3);

        auto [mem_ref_ptr, net_ref_ptr] = run_network(is_caching_test, false,
                                        input0_layout, input1_layout, input2_layout, input3_layout,
                                        input0, input1, input2, input3,
                                        use_scalar_scale_val, scale_val, use_scalar_attn_mask, attn_mask_val, is_causal);
        auto [mem_opt_ptr, net_opt_ptr] = run_network(is_caching_test, true,
                                        input0_layout, input1_layout, input2_layout, input3_layout,
                                        input0, input1, input2, input3,
                                        use_scalar_scale_val, scale_val, use_scalar_attn_mask, attn_mask_val, is_causal);

        if (is_caching_test) {
            auto inst = net_opt_ptr->get_primitive("sdpa");
            auto& sdpa_node = inst->get_node().as<scaled_dot_product_attention>();

            if (use_scalar_scale_val) {
                ASSERT_TRUE(sdpa_node.get_primitive()->scale_val.has_value());
                ASSERT_FLOAT_EQ(sdpa_node.get_primitive()->scale_val.value(), scale_val);
            }

            if (use_scalar_attn_mask) {
                ASSERT_TRUE(sdpa_node.get_primitive()->attn_mask_val.has_value());
                ASSERT_FLOAT_EQ(sdpa_node.get_primitive()->attn_mask_val.value(), attn_mask_val);
            }
        }

        cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
        {
            for (size_t idx = 0; idx < ref_data.size(); idx++) {
                ASSERT_FALSE(std::isnan(opt_data[idx]) || std::isnan(ref_data[idx])) << "NaN found at index " << idx;
            }
            auto ret = cosineSimilarity(ref_data, opt_data);
            ASSERT_GE(ret, 0.95f);
        }
    }

    static std::string
    PrintToStringParamName(const testing::TestParamInfo<sdpa_test_params>& info) {
        std::string result = "sdpa_gpu_test_" + std::to_string(info.param.head_size) + "_" +
               std::to_string(info.param.num_heads) + "_" +
               std::to_string(info.param.sequence_length_q) + "_" +
               std::to_string(info.param.sequence_length_kv) + "_" +
               std::to_string(info.param.batch);

        if (info.param.use_scalar_scale_val) {
            result += "_scale_" + std::to_string(static_cast<int>(info.param.scale_val * 1000));
        }

        if (info.param.use_scalar_attn_mask) {
            result += "_mask_" + std::to_string(static_cast<int>(info.param.attn_mask_val * 1000));
        }

        if (info.param.is_causal) {
            result += "_causal";
        }

        if (!info.param.dynamic) {
            result += "_static";
        }

        return result;
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_sdpa_gpu_test,
                         sdpa_gpu_test,
                         ::testing::Values(sdpa_test_params{64, 32, 990, 128, 2, true},                       // dynamic
                                           sdpa_test_params{64, 32, 990, 128, 2, false},                      // static
                                           sdpa_test_params{64, 32, 990, 1, 2, true},                         // dynamic
                                           sdpa_test_params{64, 32, 990, 1, 2, false},                        // static
                                           sdpa_test_params{64, 10, 77, 77, 1, true},                         // two ranks mask
                                           sdpa_test_params{64, 10, 77, 77, 1, false},                        // two ranks mask
                                           sdpa_test_params{64, 32, 128, 128, 2, true, 0.125f, false, 0.0f},  // scale_val only
                                           sdpa_test_params{64, 32, 128, 128, 2, false, 1.0f, true, 0.5f},    // attn_mask only
                                           sdpa_test_params{512, 8, 1, 1024, 2, true},
                                           sdpa_test_params{64, 32, 128, 128, 2, true, true},                 // causal dynamic
                                           sdpa_test_params{64, 32, 128, 128, 2, false, true}),               // causal static
                         sdpa_gpu_test::PrintToStringParamName);

TEST_P(sdpa_gpu_test, basic) {
    auto p = GetParam();
    execute(p);
}

TEST_P(sdpa_gpu_test, basic_caching) {
    auto p = GetParam();
    execute(p, true);
}
#endif

TEST(sdpa_gpu_custom, single_token_cond_attn_mask_clamp) {
    tests::random_generator rg; rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int head_size = 32;
    const int num_heads = 1;
    const int seq_length_q = 1;
    const int seq_length_kv = 448;
    const int batch = 1;

    layout input0_layout({batch, seq_length_q, num_heads, head_size}, data_types::f16, format::bfyx);
    layout input1_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    layout input2_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    layout input3_layout({batch, num_heads, 1, seq_length_kv}, data_types::f16, format::bfyx);

    auto input0 = engine.allocate_memory(input0_layout);
    auto input1 = engine.allocate_memory(input1_layout);
    auto input2 = engine.allocate_memory(input2_layout);
    auto input3 = engine.allocate_memory(input3_layout);


    auto fill_random = [&](memory::ptr mem) {
        auto shp = mem->get_layout().get_shape();
        size_t sz = ov::shape_size(shp);
        auto data = rg.generate_random_1d<ov::float16>(sz, -1.0f, 1.0f);
        set_values(mem, data);
    };
    fill_random(input0);
    fill_random(input1);
    fill_random(input2);

    // attention mask with first position 0, all remaining positions -inf
    {
        size_t elems = batch * num_heads * 1 * seq_length_kv;
        std::vector<ov::float16> mask(elems);
        for (int kv = 0; kv < seq_length_kv; ++kv) {
            mask[kv] = kv == 0 ? ov::float16(0.0f) : ov::float16(-std::numeric_limits<float>::infinity());
        }
        set_values(input3, mask);
    }

    topology topology;
    topology.add(input_layout("input0", input0_layout));
    topology.add(input_layout("input1", input1_layout));
    topology.add(input_layout("input2", input2_layout));
    topology.add(input_layout("input3", input3_layout));
    topology.add(scaled_dot_product_attention("sdpa", {input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3")},
                                              false, -1, {0,2,1,3}, {0,2,1,3}, {0,2,1,3}, {0,1,2,3}, {}, false));
    topology.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"sdpa", {format::type::bfyx, "sdpa_opt"}}
    }));
    auto network = get_network(engine, topology, cfg, get_test_stream_ptr(), false);
    network->set_input_data("input0", input0);
    network->set_input_data("input1", input1);
    network->set_input_data("input2", input2);
    network->set_input_data("input3", input3);
    auto output = network->execute().at("result").get_memory();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_FALSE(std::isnan(static_cast<float>(output_ptr[i])));
    }

    // With only first KV valid, output should approximate value vector at KV index 0.
    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(input2, get_test_stream());
    for (int hs = 0; hs < head_size; ++hs) {
        ASSERT_NEAR(static_cast<float>(ref_ptr[hs]), static_cast<float>(output_ptr[hs]), 1e-2f);
    }
}

TEST(sdpa_gpu_custom, scalar_placeholder_mask_matches_scale_only) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int batch = 1;
    const int seq_length_q = 4;
    const int seq_length_kv = 6;
    const int num_heads = 2;
    const int head_size = 32;
    const float scale_val = 0.35f;

    const layout q_layout({batch, seq_length_q, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout k_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout v_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout scalar_mask_layout{ov::PartialShape{}, data_types::f16, format::bfyx};

    auto q_mem = engine.allocate_memory(q_layout);
    auto k_mem = engine.allocate_memory(k_layout);
    auto v_mem = engine.allocate_memory(v_layout);
    auto scalar_mask_mem = engine.allocate_memory(scalar_mask_layout);

    auto fill_random = [&](const memory::ptr& mem) {
        const auto shape = mem->get_layout().get_shape();
        const size_t elements_num = ov::shape_size(shape);
        auto data = rg.generate_random_1d<ov::float16>(elements_num, -1.0f, 1.0f);
        set_values(mem, data);
    };

    fill_random(q_mem);
    fill_random(k_mem);
    fill_random(v_mem);
    set_values(scalar_mask_mem, {ov::float16(1.0f)});

    auto run_sdpa = [&](bool use_placeholder_mask) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        std::vector<input_info> inputs = {input_info("q"), input_info("k"), input_info("v")};
        if (use_placeholder_mask) {
            topo.add(input_layout("mask", scalar_mask_layout));
            inputs.push_back(input_info("mask"));
        }

        auto sdpa_prim = scaled_dot_product_attention("sdpa",
                                                      inputs,
                                                      false,
                                                      -1,
                                                      {0, 2, 1, 3},
                                                      {0, 2, 1, 3},
                                                      {0, 2, 1, 3},
                                                      {0, 1, 2, 3},
                                                      {},
                                                      false);
        sdpa_prim.scale_val = scale_val;

        topo.add(sdpa_prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_opt"}}}));

        auto network = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        network->set_input_data("q", q_mem);
        network->set_input_data("k", k_mem);
        network->set_input_data("v", v_mem);
        if (use_placeholder_mask) {
            network->set_input_data("mask", scalar_mask_mem);
        }

        return network->execute().at("result").get_memory();
    };

    auto output_without_mask = run_sdpa(false);
    auto output_with_placeholder_mask = run_sdpa(true);

    cldnn::mem_lock<ov::float16, mem_lock_type::read> without_mask_ptr(output_without_mask, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> with_placeholder_mask_ptr(output_with_placeholder_mask, get_test_stream());

    ASSERT_EQ(without_mask_ptr.size(), with_placeholder_mask_ptr.size());
    for (size_t i = 0; i < without_mask_ptr.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(without_mask_ptr[i]), static_cast<float>(with_placeholder_mask_ptr[i]), 1e-3f)
            << "Mismatch at index " << i
            << ", Expected : " << static_cast<float>(without_mask_ptr[i])
            << " actual : " << static_cast<float>(with_placeholder_mask_ptr[i])
            << std::endl;
    }
}

// Test that an explicit causal attention mask produces the same result as is_causal=true.
static void run_sdpa_causal_mask(int batch, int q_num_heads, int kv_num_heads,
                                     int seq_q, int seq_kv, int head_size) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto q_data = rg.generate_random_1d<ov::float16>(
        static_cast<size_t>(batch) * q_num_heads * seq_q * head_size, -1.0f, 1.0f);
    auto k_data = rg.generate_random_1d<ov::float16>(
        static_cast<size_t>(batch) * kv_num_heads * seq_kv * head_size, -1.0f, 1.0f);
    auto v_data = rg.generate_random_1d<ov::float16>(
        static_cast<size_t>(batch) * kv_num_heads * seq_kv * head_size, -1.0f, 1.0f);

    // Build causal attention mask: shape [1, 1, seq_q, seq_kv]
    // 0 for valid positions (row >= col offset), -inf for masked positions.
    const size_t mask_size = static_cast<size_t>(seq_q) * seq_kv;
    std::vector<ov::float16> mask_data(mask_size);
    const int col_offset = seq_kv - seq_q;
    for (int r = 0; r < seq_q; ++r) {
        for (int c = 0; c < seq_kv; ++c) {
            if (c <= r + col_offset) {
                mask_data[r * seq_kv + c] = ov::float16(0.0f);
            } else {
                mask_data[r * seq_kv + c] = ov::float16(-INFINITY);
            }
        }
    }

    const layout q_layout({batch, q_num_heads, seq_q, head_size}, data_types::f16, format::bfyx);
    const layout kv_layout({batch, kv_num_heads, seq_kv, head_size}, data_types::f16, format::bfyx);
    const layout mask_layout({1, 1, seq_q, seq_kv}, data_types::f16, format::bfyx);

    const layout q_dyn_layout({batch, q_num_heads, -1, head_size}, data_types::f16, format::bfyx);
    const layout kv_dyn_layout({batch, kv_num_heads, -1, head_size}, data_types::f16, format::bfyx);
    const layout mask_dyn_layout({1, 1, -1, -1}, data_types::f16, format::bfyx);

    auto q_mem = engine.allocate_memory(q_layout);
    auto k_mem = engine.allocate_memory(kv_layout);
    auto v_mem = engine.allocate_memory(kv_layout);
    auto mask_mem = engine.allocate_memory(mask_layout);
    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(v_mem, v_data);
    set_values(mask_mem, mask_data);

    // --- Golden reference: is_causal=false, explicit mask as 4th input, static shapes ---
    auto make_ref_output = [&]() {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_layout));
        topo.add(input_layout("v", kv_layout));
        topo.add(input_layout("mask", mask_layout));
        auto prim = scaled_dot_product_attention("sdpa",
                                                 {input_info("q"), input_info("k"), input_info("v"), input_info("mask")},
                                                 false, -1,
                                                 {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3},
                                                 {}, false);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("mask", mask_mem);
        return net->execute().at("result").get_memory();
    };

    // --- Optimized path: is_causal=true, no mask input, dynamic shapes ---
    auto make_opt_output = [&]() {
        topology topo;
        topo.add(input_layout("q", q_dyn_layout));
        topo.add(input_layout("k", kv_dyn_layout));
        topo.add(input_layout("v", kv_dyn_layout));
        auto prim = scaled_dot_product_attention("sdpa",
                                                 {input_info("q"), input_info("k"), input_info("v")},
                                                 true, -1,
                                                 {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3},
                                                 {}, false);
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        return net->execute().at("result").get_memory();
    };

    auto ref_mem = make_ref_output();
    auto opt_mem = make_opt_output();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(ref_mem, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_ptr(opt_mem, get_test_stream());

    ASSERT_EQ(ref_ptr.size(), opt_ptr.size());
    for (size_t i = 0; i < ref_ptr.size(); ++i) {
        ASSERT_FALSE(std::isnan(static_cast<float>(ref_ptr[i]))) << "NaN in explicit mask output at index " << i;
        ASSERT_FALSE(std::isnan(static_cast<float>(opt_ptr[i]))) << "NaN in is_causal output at index " << i;
    }

    const float sim = cosineSimilarity(ref_ptr, opt_ptr);
    ASSERT_GE(sim, 0.99f) << "explicit mask vs is_causal cosine similarity too low: " << sim;
}

TEST(sdpa_gpu_causal_mask, prefill_40q_40kv_512seq) {
    run_sdpa_causal_mask(1, 40, 40, 512, 512, 128);
}

TEST(sdpa_gpu_causal_mask, decode_40q_40kv_512seq) {
    run_sdpa_causal_mask(1, 40, 40, 1, 512, 128);
}

TEST(sdpa_gpu_causal_mask, prefill_40q_10kv_512seq) {
    run_sdpa_causal_mask(1, 40, 10, 512, 512, 128);
}

TEST(sdpa_gpu_causal_mask, decode_40q_10kv_444seq) {
    run_sdpa_causal_mask(1, 40, 10, 1, 444, 128);
}

TEST(sdpa_gpu_causal_mask, decode_32q_8kv_1024seq) {
    run_sdpa_causal_mask(1, 32, 8, 1, 1024, 128);
}


// ---------------------------------------------------------------------------
// Compressed (int8 / int4) KV-cache SDPA tests.
//
// The optimized SDPA kernel (sdpa_opt) dequantizes the KV cache on read using
// the asymmetric formula `deq = (q - zp) * scale`, where a single (scale, zp)
// pair is shared by all head_size channels of a given (batch, head, token) and
// stored interleaved as [scale, zp] (InterleavedScalesZP) in fp16. INT4 packs
// two unsigned nibbles per byte: low nibble -> even head dim, high nibble -> odd
// head dim.
//
// These tests build a `scaled_dot_product_attention` primitive with
// is_kv_compressed=true directly and feed host-quantized KV plus the matching
// scale/zp buffers to the forced `sdpa_opt` path. The golden reference is an
// independent float attention (uncompressed, forced `sdpa_ref`) fed the exact
// host-dequantized KV, so the two networks operate on identical effective KV
// and any divergence isolates the kernel's dequant + attention math.
// ---------------------------------------------------------------------------
struct kv_quant_result {
    std::vector<int8_t> packed;             // quantized (INT4: two nibbles per byte)
    std::vector<ov::float16> dequantized;   // (q - zp) * scale, full head_size
    std::vector<ov::float16> scales_zp;     // interleaved [scale, zp] per (b, head, token)
};

// Asymmetric per-(batch, head, token) quantization over the whole head_size dim.
// Layout of `src` is bfyx {batch, seq, heads, head_size} (matches transpose {0,2,1,3}).
// Load a raw little-endian binary dump into a typed vector (no header parsing).
template <typename T>
static std::vector<T> load_bin_as(const std::string& path) {
    const auto bytes = ov::util::load_binary(path);
    OPENVINO_ASSERT(!bytes.empty(), "Failed to load (missing/empty) binary file: ", path);
    OPENVINO_ASSERT(bytes.size() % sizeof(T) == 0, "Binary file size is not a multiple of element size: ", path);
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

// When seq_major is true the source is laid out [batch, seq, heads, head_size]; when false it is
// [batch, heads, seq, head_size].
//
// Two quantization strategies are supported:
//   * Asymmetric (symmetric=false): per-(batch, head, token) over the whole head_size dim.
//     Produces an interleaved [scale, zp] buffer ([batch, heads, seq, 2]) in scales_zp.
//   * Symmetric  (symmetric=true):  one scale per (batch, head, channel) shared across the
//     sequence dimension (deq = q * scale, no zero-point). provided_scale, when non-null,
//     supplies these per-channel scales laid out [batch, heads, head_size]; otherwise the
//     per-channel scale is derived from the max-abs over all sequence positions.
static kv_quant_result quantize_kv_per_token(const std::vector<ov::float16>& src,
                                             int batch, int seq, int heads, int head_size,
                                             int bit_width, bool seq_major = true,
                                             const std::vector<ov::float16>* provided_scale = nullptr,
                                             bool symmetric = false) {
    // Symmetric uses a signed range centered on zero (no zero-point); asymmetric int4 uses
    // unsigned nibbles [0, 15] with a zero-point.
    const int q_min = symmetric ? -(bit_width == 4 ? 7 : 127) : (bit_width == 4 ? 0 : -128);
    const int q_max = symmetric ? (bit_width == 4 ? 7 : 127) : (bit_width == 4 ? 15 : 127);
    const int packed_hs = bit_width == 4 ? head_size / 2 : head_size;

    kv_quant_result r;
    r.packed.assign(static_cast<size_t>(batch) * seq * heads * packed_hs, 0);
    r.dequantized.assign(static_cast<size_t>(batch) * seq * heads * head_size, ov::float16(0.0f));
    r.scales_zp.assign(static_cast<size_t>(batch) * heads * seq * 2, ov::float16(0.0f));

    const auto elem_base = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * head_size
                         : ((static_cast<size_t>(b) * heads + h) * seq + s) * head_size;
    };
    const auto packed_base_of = [&](int b, int s, int h) -> size_t {
        return seq_major ? ((static_cast<size_t>(b) * seq + s) * heads + h) * packed_hs
                         : ((static_cast<size_t>(b) * heads + h) * seq + s) * packed_hs;
    };
    const auto pack_value = [&](size_t packed_base, int d, int q) {
        if (bit_width == 4) {
            // low nibble = even head dim, high nibble = odd head dim
            auto& byte = r.packed[packed_base + d / 2];
            if (d % 2 == 0)
                byte = static_cast<int8_t>((byte & 0xF0) | (q & 0x0F));
            else
                byte = static_cast<int8_t>((byte & 0x0F) | ((q & 0x0F) << 4));
        } else {
            r.packed[packed_base + d] = static_cast<int8_t>(q);
        }
    };

    if (symmetric) {
        // Per-channel symmetric quantization: one scale per (batch, head, channel), shared
        // across the whole sequence dimension (deq = q * scale, no zero-point).
        std::vector<float> channel_scale(static_cast<size_t>(batch) * heads * head_size, 0.0f);
        if (provided_scale != nullptr) {
            for (size_t i = 0; i < channel_scale.size(); ++i)
                channel_scale[i] = static_cast<float>((*provided_scale)[i]);
        } else {
            for (int b = 0; b < batch; ++b)
                for (int h = 0; h < heads; ++h)
                    for (int d = 0; d < head_size; ++d) {
                        float max_abs = 0.0f;
                        for (int s = 0; s < seq; ++s)
                            max_abs = std::max(max_abs, std::abs(static_cast<float>(src[elem_base(b, s, h) + d])));
                        channel_scale[(static_cast<size_t>(b) * heads + h) * head_size + d] =
                            max_abs / static_cast<float>(q_max);
                    }
        }
        for (auto& sc : channel_scale)
            if (sc == 0.0f)
                sc = 1.0f;  // degenerate (constant) channel: avoid div-by-zero

        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < seq; ++s) {
                for (int h = 0; h < heads; ++h) {
                    const size_t base = elem_base(b, s, h);
                    const size_t packed_base = packed_base_of(b, s, h);
                    const size_t scale_base = (static_cast<size_t>(b) * heads + h) * head_size;
                    for (int d = 0; d < head_size; ++d) {
                        const float scale = channel_scale[scale_base + d];
                        const float v = static_cast<float>(src[base + d]);
                        int q = static_cast<int>(std::lround(v / scale));
                        q = std::max(q_min, std::min(q_max, q));
                        r.dequantized[base + d] = ov::float16(static_cast<float>(q) * scale);
                        pack_value(packed_base, d, q);
                    }
                }
            }
        }
        return r;
    }

    if (!symmetric){
        assert(false && "Asymmetric quantization not implemented yet");
    }
    
    return r;
}

static void run_compressed_kv_sdpa_test(int bit_width) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int batch = 1;
    const int num_heads = 2;
    const int seq_q = 16;
    const int seq_kv = 16;
    const int head_size = 64;
    const int packed_hs = bit_width == 4 ? head_size / 2 : head_size;
    const float scale_val = 1.0f / std::sqrt(static_cast<float>(head_size));

    // Random Q and original (float) K/V. bfyx physical layout {batch, seq, heads, head_size}.
    auto q_data = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch) * seq_q * num_heads * head_size, -1.0f, 1.0f);
    auto k_orig = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch) * seq_kv * num_heads * head_size, -1.0f, 1.0f);
    auto v_orig = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch) * seq_kv * num_heads * head_size, -1.0f, 1.0f);

    auto k_q = quantize_kv_per_token(k_orig, batch, seq_kv, num_heads, head_size, bit_width);
    auto v_q = quantize_kv_per_token(v_orig, batch, seq_kv, num_heads, head_size, bit_width);

    const layout q_layout({batch, seq_q, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout kv_deq_layout({batch, seq_kv, num_heads, head_size}, data_types::f16, format::bfyx);
    const layout kv_packed_layout({batch, seq_kv, num_heads, packed_hs}, data_types::i8, format::bfyx);
    const layout comp_layout({batch, num_heads, seq_kv, 2}, data_types::f16, format::bfyx);

    auto q_mem = engine.allocate_memory(q_layout);
    set_values(q_mem, q_data);

    // --- Golden reference: uncompressed float attention on host-dequantized KV (sdpa_ref) ---
    auto make_ref_output = [&]() {
        auto k_mem = engine.allocate_memory(kv_deq_layout);
        auto v_mem = engine.allocate_memory(kv_deq_layout);
        set_values(k_mem, k_q.dequantized);
        set_values(v_mem, v_q.dequantized);

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_deq_layout));
        topo.add(input_layout("v", kv_deq_layout));
        auto prim = scaled_dot_product_attention("sdpa",
                                                 {input_info("q"), input_info("k"), input_info("v")},
                                                 false, -1, {0, 2, 1, 3}, {0, 2, 1, 3}, {0, 2, 1, 3}, {0, 1, 2, 3}, {}, false);
        prim.scale_val = scale_val;
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_ref"}}}));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        return net->execute().at("result").get_memory();
    };

    // --- Compressed path: quantized KV + interleaved scale/zp on the optimized kernel (sdpa_opt) ---
    auto make_opt_output = [&]() {
        auto k_mem = engine.allocate_memory(kv_packed_layout);
        auto v_mem = engine.allocate_memory(kv_packed_layout);
        auto k_comp_mem = engine.allocate_memory(comp_layout);
        auto v_comp_mem = engine.allocate_memory(comp_layout);
        set_values(k_mem, k_q.packed);
        set_values(v_mem, v_q.packed);
        set_values(k_comp_mem, k_q.scales_zp);
        set_values(v_comp_mem, v_q.scales_zp);

        scaled_dot_product_attention::QuantizationAttributes qa;
        qa.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        qa.quantization_dt = ov::element::i8;
        qa.scale_dt = ov::element::f16;
        qa.zp_dt = ov::element::f16;
        qa.group_sizes = {1, 1, 1, UINT64_MAX};
        qa.scales_zp_output_order = {0, 1, 2, 3};
        qa.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", kv_packed_layout));
        topo.add(input_layout("v", kv_packed_layout));
        topo.add(input_layout("k_scale", comp_layout));
        topo.add(input_layout("v_scale", comp_layout));

        // Inputs: Q, K, V, key_scales, value_scales (no separate zp for InterleavedScalesZP).
        auto prim = scaled_dot_product_attention("sdpa",
                                                 {input_info("q"), input_info("k"), input_info("v"),
                                                  input_info("k_scale"), input_info("v_scale")},
                                                 false, -1, {0, 2, 1, 3}, {0, 2, 1, 3}, {0, 2, 1, 3}, {0, 1, 2, 3}, qa, true);
        prim.scale_val = scale_val;
        topo.add(prim);
        topo.add(reorder("result", input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::hint::kv_cache_precision(bit_width == 4 ? ov::element::i4 : ov::element::i8));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"sdpa", {format::type::bfyx, "sdpa_opt"}}}));

        auto net = get_network(engine, topo, cfg, get_test_stream_ptr(), false);
        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("k_scale", k_comp_mem);
        net->set_input_data("v_scale", v_comp_mem);
        return net->execute().at("result").get_memory();
    };

    auto ref_mem = make_ref_output();
    auto opt_mem = make_opt_output();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_ptr(ref_mem, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_ptr(opt_mem, get_test_stream());

    ASSERT_EQ(ref_ptr.size(), opt_ptr.size());
    for (size_t i = 0; i < opt_ptr.size(); ++i) {
        ASSERT_FALSE(std::isnan(static_cast<float>(opt_ptr[i]))) << "NaN in compressed output at index " << i;
    }
    const float sim = cosineSimilarity(ref_ptr, opt_ptr);
    ASSERT_GE(sim, 0.95f) << "bit_width=" << bit_width << " cosine similarity too low: " << sim;
}

TEST(sdpa_gpu_compressed_kv, int8) {
    run_compressed_kv_sdpa_test(8);
}

TEST(sdpa_gpu_compressed_kv, int4) {
    run_compressed_kv_sdpa_test(4);
}


} // namespace
