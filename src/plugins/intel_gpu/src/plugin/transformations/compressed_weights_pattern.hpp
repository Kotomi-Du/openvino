// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/op/optional.hpp"

using namespace ov::pass::pattern;
using ov::pass::operator|;
#define FC_COMPRESSED_WEIGHT_PATTERN\
        auto reshape_squeeze = [](const ov::Output<ov::Node>& output) {\
            auto in_ps = output.get_node()->get_input_partial_shape(0);\
            auto out_ps = output.get_node()->get_output_partial_shape(0);\
            return in_ps.rank().is_static() && out_ps.rank().is_static() &&\
                   ((in_ps.size() == 3 && out_ps.size() == 2) || (in_ps.size() == 4 && out_ps.size() == 3));\
        };\
        \
        auto weights_m = wrap_type<ov::op::v0::Constant, ov::op::v0::Parameter>(\
            type_matches_any({ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4,\
                              ov::element::u2, ov::element::f8e4m3, ov::element::f8e5m2,\
                              ov::element::f4e2m1}));\
        auto weights_reshape_m = ov::pass::pattern::optional<ov::op::v1::Reshape>({weights_m, any_input()});\
        auto convert_m = wrap_type<ov::op::v0::Convert>({weights_reshape_m});\
        auto decompressed_weights_m =\
            ov::pass::pattern::optional<ov::op::v1::Reshape>({convert_m, any_input()});\
\
        auto zp_m = wrap_type<ov::op::v0::Constant, ov::op::v0::Parameter>();\
        auto zp_convert_m = ov::pass::pattern::optional<ov::op::v0::Convert>({zp_m});\
        auto zp_reshape_m =\
            ov::pass::pattern::optional<ov::op::v1::Reshape>({zp_convert_m, any_input()});\
        auto zp_decompression_m = ov::pass::pattern::optional<ov::op::v0::Convert>({zp_reshape_m});\
        auto subtract_m =\
            ov::pass::pattern::optional<ov::op::v1::Subtract>({decompressed_weights_m, zp_decompression_m});\
\
        auto scale_m = wrap_type<ov::op::v0::Constant, ov::op::v0::Parameter>();\
        auto scale_convert_m = ov::pass::pattern::optional<ov::op::v0::Convert>({scale_m});\
        auto scale_reshape_m =\
            ov::pass::pattern::optional<ov::op::v1::Reshape>({scale_convert_m, any_input()});\
        auto scale_decompression_m = ov::pass::pattern::optional<ov::op::v0::Convert>({scale_reshape_m});\
        auto mul_m = wrap_type<ov::op::v1::Multiply>({subtract_m, scale_decompression_m});\
\
        /* No transpose: Multiply, Reshape, Convert(Reshape), or Multiply(Reshape). */\
        auto reshape_const_m = wrap_type<ov::op::v0::Constant>();\
        auto reshape_m = wrap_type<ov::op::v1::Reshape>({mul_m, reshape_const_m}, reshape_squeeze);\
        auto convert_reshape_m = ov::pass::pattern::optional<ov::op::v0::Convert>({reshape_m});\
        auto mul2_const_m = wrap_type<ov::op::v0::Constant>();\
        auto mul2_m = ov::pass::pattern::optional<ov::op::v1::Multiply>({reshape_m, mul2_const_m});\
        auto no_transpose_m = mul_m | convert_reshape_m | mul2_m;\
\
        /* Transpose after decompression, with an optional preceding Reshape. */\
        auto transpose_const_m = wrap_type<ov::op::v0::Constant>();\
        auto transpose_after_reshape_input_m = reshape_m | mul_m;\
        auto transpose_after_reshape_m =\
                wrap_type<ov::op::v1::Transpose>({transpose_after_reshape_input_m, transpose_const_m});\
\
        /* Transpose before the final Reshape. */\
        auto transpose_before_reshape_input_m =\
                wrap_type<ov::op::v1::Transpose>({mul_m, wrap_type<ov::op::v0::Constant>()});\
        auto transpose_before_reshape_m = wrap_type<ov::op::v1::Reshape>(\
                {transpose_before_reshape_input_m, transpose_const_m}, reshape_squeeze);\
\
        auto compressed_weights_input_m =\
            no_transpose_m | transpose_after_reshape_m | transpose_before_reshape_m;
