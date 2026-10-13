// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Qwen3.8-Flash-Next vision encoder (the checkpoint's `model.visual`, BF16): images and video frame pairs → embeddings of the image tokens.
//
//   patches [N, 3·2·16·16] (one temporal patch = an image repeated twice, or two video frames; rows in 2×2 merge-block order)
//     → patch embedding (the 3D convolution as one linear map, with bias) + learned position embedding (48×48 grid, bilinearly resampled
//       to gh × gw with align-corners sample points, permuted into merge-block order) → 27 blocks of
//         x += proj(attention(rope2d(q), rope2d(k), v))     [LayerNorm in front, non-causal over the N patches, 16 heads × 72]
//         x += fc2(gelu_tanh(fc1(LayerNorm(x))))
//     → merger: LayerNorm per patch, 4 patches of a merge block concatenated (4608) → fc1 → GELU (erf) → fc2 → [N/4, 2560]
//   Rotary: axial 2D, θ 10,000, 18 frequencies 10000^(−2j/36); dims [0,18) row · [18,36) column, repeated for [36,72) (rotate-half
//   over the whole head of 72).
//   Behaviour follows the model's published configuration (config.json vision_config, preprocessor_config.json); the code is this
//   project's own (adapted from hive/glm/glm_vision.h).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "hive/cublas_ops.h"
#include "hive/devbuf.h"
#include "hive/qwen/qwen_model.h"

namespace hive::qwen {

class QwenVision {
 public:
  explicit QwenVision(QwenModel& m);
  static constexpr int kPatchDim = 3 * 2 * 16 * 16;  // 1536: (channel, frame, py, px) = patch_embed.proj.weight [1152, 3, 2, 16, 16] flattened
  // device memory one encode of n patches needs beyond the weights (activations + one attention tile); allocated inside encode
  static size_t scratch_bytes(int n_patches);
  size_t weight_bytes() const { return bytes_; }
  // One temporal patch: gh × gw patches (both even), `patches` device bf16 [gh·gw, 1536] → `out` device bf16 [gh·gw/4, 2560].
  void encode(const __nv_bfloat16* patches, int gh, int gw, __nv_bfloat16* out, cudaStream_t st);

 private:
  struct Block { DevBuf n1w, n1b, n2w, n2b, qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b; };
  DevBuf load(QwenModel& m, const std::string& name, size_t expect_elems);
  std::vector<Block> blocks_;
  DevBuf patch_w_, patch_b_, mn_w_, mn_b_, m1_w_, m1_b_, m2_w_, m2_b_, inv_freq_;
  DevBuf pos_;                      // learned position grid [side·side, dim] bf16 (resampled per encode)
  Blas blas_;
  size_t bytes_ = 0;
  int dim_ = 1152, heads_ = 16, inter_ = 4304, out_ = 2560, depth_ = 27, side_ = 48;
  float eps_ = 1e-6f;
};

}  // namespace hive::qwen
