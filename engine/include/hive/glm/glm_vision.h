// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// GLM-5.3-Flash vision encoder (the checkpoint's `model.visual`, BF16): images and video frames → embeddings of the image tokens.
//
//   patches [N, 3·2·14·14] (one temporal patch = an image repeated twice, or two video frames; rows in 2×2 merge-block order)
//     → patch embedding (the 3D convolution as one linear map) → 24 blocks of
//         x += proj(attention(rope2d(rmsnorm_head(q)), rope2d(rmsnorm_head(k)), v))   [non-causal over the N patches]
//         x += down(silu(min(gate, 10)) · clamp(up, ±10))
//     → rmsnorm → 2×2 downsample (convolution as a linear map over each merge block) → merger
//         (linear → LayerNorm → GELU → clamped SwiGLU) → [N/4, 4096]
//   Rotary: axial 2D, θ 10,000, the same 16 frequencies for the row and the column index; dims [0,16) row · [16,32) column,
//   repeated for [32,64) (rotate-half over the whole head of 64).
//   Behaviour follows the model's published configuration (config.json vision_config, processor_config.json); the code is
//   this project's own.
#pragma once
#include <cstdint>
#include <vector>

#include "hive/cublas_ops.h"
#include "hive/devbuf.h"
#include "hive/glm/glm_model.h"

namespace hive::glm {

class GlmVision {
 public:
  explicit GlmVision(GlmModel& m);
  // One temporal patch: gh × gw patches (both even), `patches` device bf16 [gh·gw, 1176] → `out` device bf16 [gh·gw/4, 4096].
  void encode(const bf16* patches, int gh, int gw, bf16* out, cudaStream_t st);
  size_t weight_bytes() const { return bytes_; }
  // device memory one encode of n patches needs beyond the weights (activations + one attention tile)
  static size_t scratch_bytes(int n);
  static constexpr int kPatchDim = 3 * 2 * 14 * 14;

 private:
  struct Block {
    DevBuf norm1, norm2, qkv_w, qkv_b, q_norm, k_norm, proj_w, proj_b, gate_w, gate_b, up_w, up_b, down_w, down_b;
  };
  DevBuf load(GlmModel& m, const std::string& name, size_t expect_elems);
  std::vector<Block> blocks_;
  DevBuf patch_w_, patch_b_, post_norm_, down_w_, down_b_;
  DevBuf m_proj_, m_ln_w_, m_ln_b_, m_gate_, m_up_, m_down_;
  DevBuf inv_freq_;
  Blas blas_;
  size_t bytes_ = 0;
  int dim_ = 1024, heads_ = 16, inter_ = 4096, out_ = 4096, pinter_ = 10240, depth_ = 24;
  float eps_ = 1e-5f, limit_ = 10.f;
};

}  // namespace hive::glm
