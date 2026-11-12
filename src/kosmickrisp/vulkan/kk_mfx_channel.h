/*
 * SPDX-License-Identifier: MIT
 */

/* The private MetalFX upscale channel over VK_NVX_binary_import.
 *
 * Shared with the Windows-side runtimes that drive it, so it only uses plain C types.
 *
 * - vkCreateCuModuleNVX with data that starts with KK_MFX_MODULE_MAGIC creates
 *   the upscaler module. Any other data is refused (no CUDA on Apple GPUs).
 * - vkCreateCuFunctionNVX with pName KK_MFX_FUNCTION_UPSCALE or
 *   KK_MFX_FUNCTION_INTERPOLATE on it.
 * - vkCmdCuLaunchKernelNVX with that function, pParams pointing at a
 *   struct kk_mfx_upscale_params (or kk_mfx_interpolate_params) and
 *   paramCount = sizeof(that struct), in bytes. (This deliberately differs from CUDA's meaning: Mesa's command
 *   replay copies paramCount bytes from pParams, so the block survives
 *   re-recording.) The grid and block sizes are ignored.
 *
 * Images are VK_NVX_image_view_handle 64-bit handles (vkGetImageViewHandle64NVX).
 * Values keep their DLSS meaning; the driver maps them onto MetalFX.
 */

#ifndef KK_MFX_CHANNEL_H
#define KK_MFX_CHANNEL_H 1

#include <stddef.h>
#include <stdint.h>

#define KK_MFX_MODULE_MAGIC "KKMETALFX1"
#define KK_MFX_FUNCTION_UPSCALE "kk_mfx_upscale"
#define KK_MFX_FUNCTION_INTERPOLATE "kk_mfx_interpolate"

#define KK_MFX_PARAMS_MAGIC 0x584D4B4Bu /* "KKMX" */
/* Version 2 appended max_render_width/height and sharpness; the driver also
 * takes version 1 blocks (paramCount = KK_MFX_PARAMS_V1_SIZE). */
#define KK_MFX_PARAMS_VERSION 2u

/* DLSS feature-create flags (NVSDK_NGX_DLSS_Feature_Flags) plus reset */
enum kk_mfx_flags {
   KK_MFX_FLAG_IS_HDR = 1u << 0,
   KK_MFX_FLAG_MV_LOW_RES = 1u << 1,
   KK_MFX_FLAG_MV_JITTERED = 1u << 2,
   KK_MFX_FLAG_DEPTH_INVERTED = 1u << 3,
   KK_MFX_FLAG_DO_SHARPENING = 1u << 5,
   KK_MFX_FLAG_AUTO_EXPOSURE = 1u << 6,
   KK_MFX_FLAG_RESET = 1u << 31,
};

struct kk_mfx_upscale_params {
   uint32_t magic;   /* KK_MFX_PARAMS_MAGIC */
   uint32_t version; /* KK_MFX_PARAMS_VERSION */

   /* Image view handles; exposure may be 0 */
   uint64_t color;
   uint64_t depth;
   uint64_t motion;
   uint64_t output;
   uint64_t exposure;

   /* Render size this frame: the part of color, depth and motion that holds
    * it. With dynamic resolution it changes from frame to frame, up to
    * max_render_width/height. */
   uint32_t render_width;
   uint32_t render_height;
   /* Output size, written at output_x/y */
   uint32_t output_width;
   uint32_t output_height;
   uint32_t color_x, color_y;
   uint32_t depth_x, depth_y;
   uint32_t motion_x, motion_y;
   uint32_t output_x, output_y;

   /* DLSS Jitter.Offset: the offset applied to the projection, in render pixels */
   float jitter_x, jitter_y;
   /* DLSS MV.Scale: multiplies the motion texture into render pixels */
   float mv_scale_x, mv_scale_y;
   /* DLSS.Pre.Exposure (1 if unused) */
   float pre_exposure;

   uint32_t flags; /* enum kk_mfx_flags */

   /* Version 2 */
   /* The largest render size of the feature (DLSS create Width/Height).
    * Informational: the driver sizes a scaler for each render size. */
   uint32_t max_render_width;
   uint32_t max_render_height;
   /* Sharpening after the upscale, 0 (off) to 1 (DLSS Sharpness when the
    * DoSharpening flag is set) */
   float sharpness;
};

#define KK_MFX_PARAMS_V1_SIZE offsetof(struct kk_mfx_upscale_params, max_render_width)

/* Frame generation (DLSS-G): a frame between the previous color and this one */
#define KK_MFX_INTERPOLATE_MAGIC 0x494D4B4Bu /* "KKMI" */
#define KK_MFX_INTERPOLATE_VERSION 1u

enum kk_mfx_interpolate_flags {
   KK_MFX_INTERPOLATE_RESET = 1u << 0,
   KK_MFX_INTERPOLATE_DEPTH_INVERTED = 1u << 1,
   /* ui holds the UI on its own (color is HUD-less); otherwise the UI is
    * already in color */
   KK_MFX_INTERPOLATE_UI_SEPARATE = 1u << 2,
};

struct kk_mfx_interpolate_params {
   uint32_t magic;   /* KK_MFX_INTERPOLATE_MAGIC */
   uint32_t version; /* KK_MFX_INTERPOLATE_VERSION */

   /* Image view handles. prev_color may be 0: the driver keeps a copy of
    * each frame's color for the next one. ui may be 0. output_real, if not 0,
    * receives a copy of color (DLSS-G's real-frame output). */
   uint64_t color;
   uint64_t prev_color;
   uint64_t depth;
   uint64_t motion;
   uint64_t ui;
   uint64_t output;
   uint64_t output_real;

   /* Color and output size; color is read at color_x/y, output written at
    * output_x/y */
   uint32_t width, height;
   uint32_t color_x, color_y;
   uint32_t output_x, output_y;
   /* Depth and motion size (the render size) */
   uint32_t input_width, input_height;

   /* Multiplies the motion texture into pixels of the depth/motion size;
    * vectors point from the current to the previous position */
   float mv_scale_x, mv_scale_y;
   /* Projection jitter, in render pixels */
   float jitter_x, jitter_y;
   /* Seconds between the previous frame and this one (0: unknown) */
   float delta_time;
   float near_plane, far_plane;
   /* Vertical field of view in radians, view width / height */
   float fov;
   float aspect_ratio;

   uint32_t flags; /* enum kk_mfx_interpolate_flags */
};

#endif /* KK_MFX_CHANNEL_H */
