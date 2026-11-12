/*
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_MFX_H
#define KK_MFX_H 1

#include "kk_private.h"

#include "util/list.h"
#include "util/simple_mtx.h"

#include "kosmickrisp/bridge/mtl_types.h"

struct kk_cmd_buffer;
struct kk_device;
struct kk_image_view;
struct kk_mfx_upscale_params;
struct kk_mfx_interpolate_params;
struct kk_mfx_interpolator;

/* MetalFX temporal scalers, one per (formats, sizes, flags), kept for the
 * device's lifetime. See kk_mfx.c. */
struct kk_mfx_state {
   simple_mtx_t mutex;
   /* Most recently used first, at most KK_MFX_MAX_SCALERS (kk_mfx.c) */
   struct list_head scalers;
   uint32_t scaler_count;
   /* Evicted scalers waiting until no recorded work can still use them */
   struct list_head retired;
   uint64_t upscale_count;
   uint32_t created_count;
   /* Sharpening pipelines per strength step and output offset */
   struct list_head sharpeners;
   /* Binds the sharpening pass's two textures */
   mtl_argument_table *texture_table;
   /* The last upscale's scaler and output texture, to reset history when a
    * render size change switches scalers */
   struct kk_mfx_scaler *last_scaler;
   mtl_texture *last_output;
   /* Frame generation: one interpolator, recreated when its inputs change */
   struct kk_mfx_interpolator *interpolator;
   /* Interpolators replaced while recorded work may still use them */
   struct list_head retired_interpolators;
};

void kk_mfx_init(struct kk_device *dev);
void kk_mfx_finish(struct kk_device *dev);

/* Resolves a VK_NVX_image_view_handle 64-bit handle, NULL if it isn't one of
 * this device's image views */
struct kk_image_view *kk_mfx_view_from_handle(struct kk_device *dev,
                                              uint64_t handle);

/* Records a MetalFX upscale (the kk_mfx_channel.h launch) */
void kk_mfx_cmd_upscale(struct kk_cmd_buffer *cmd,
                        const struct kk_mfx_upscale_params *params);

/* Records a MetalFX frame interpolation (the kk_mfx_channel.h launch) */
void kk_mfx_cmd_interpolate(struct kk_cmd_buffer *cmd,
                            const struct kk_mfx_interpolate_params *params);

#endif /* KK_MFX_H */
