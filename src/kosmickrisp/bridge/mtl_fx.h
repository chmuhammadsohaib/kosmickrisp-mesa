/*
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_FX_H
#define MTL_FX_H 1

#include "mtl_format.h"
#include "mtl_types.h"

#include <inttypes.h>
#include <stdbool.h>

typedef void mtl_fx_temporal_scaler;

/* Fixed when the scaler is created */
struct mtl_fx_temporal_scaler_desc {
   enum mtl_pixel_format color_format;
   enum mtl_pixel_format depth_format;
   enum mtl_pixel_format motion_format;
   enum mtl_pixel_format output_format;
   uint32_t input_width;
   uint32_t input_height;
   uint32_t output_width;
   uint32_t output_height;
   bool auto_exposure;
   /* Frames may cover less than input_width/height (dynamic resolution),
    * with output/input ratios between these scales */
   bool input_content_properties;
   float input_content_min_scale;
   float input_content_max_scale;
   /* Motion vectors at output resolution instead of input resolution */
   bool output_resolution_motion;
   bool jittered_motion;
};

/* Set for every frame, in MetalFX's own conventions */
struct mtl_fx_temporal_scaler_frame {
   mtl_texture *color;
   mtl_texture *depth;
   mtl_texture *motion;
   mtl_texture *output;
   mtl_texture *exposure; /* may be NULL */
   /* Part of the inputs holding this frame */
   uint32_t content_width, content_height;
   uint32_t color_x, color_y;
   uint32_t depth_x, depth_y;
   uint32_t motion_x, motion_y;
   float jitter_x, jitter_y;
   float motion_scale_x, motion_scale_y;
   float pre_exposure;
   bool reset;
   bool depth_reversed;
};

bool mtl_fx_temporal_scaler_supported(mtl_device *device);
/* Largest output/input ratio the temporal scaler supports on this device */
float mtl_fx_temporal_scaler_max_scale(mtl_device *device);
mtl_fx_temporal_scaler *
mtl_new_fx_temporal_scaler(mtl_device *device, mtl_compiler *compiler,
                           const struct mtl_fx_temporal_scaler_desc *desc);
/* A private texture the scaler can write (MetalFX requires private output) */
mtl_texture *mtl_fx_new_output_texture(mtl_device *device,
                                       mtl_fx_temporal_scaler *scaler);
/* The MTLTextureUsage the scaler needs on its inputs */
uint32_t mtl_fx_input_usage(mtl_fx_temporal_scaler *scaler);
void mtl_fx_temporal_scaler_encode(
   mtl_fx_temporal_scaler *scaler, mtl_command_buffer *cmd_buf,
   const struct mtl_fx_temporal_scaler_frame *frame);

/* An argument table for a compute pass that binds two textures */
mtl_argument_table *mtl_fx_new_texture_table(mtl_device *device);
/* Runs `pipeline` over width x height threads in its own compute encoder,
 * with src at texture 0 and dst at texture 1, ordered after all earlier work
 * and before all later work in the command buffer */
void mtl_fx_encode_texture_pass(mtl_command_buffer *cmd_buf,
                                mtl_compute_pipeline_state *pipeline,
                                mtl_argument_table *table, mtl_texture *src,
                                mtl_texture *dst, uint32_t width,
                                uint32_t height);

enum mtl_pixel_format mtl_texture_get_pixel_format(mtl_texture *texture);
uint32_t mtl_texture_get_width(mtl_texture *texture);
uint32_t mtl_texture_get_height(mtl_texture *texture);
uint32_t mtl_texture_get_usage(mtl_texture *texture);

#endif /* MTL_FX_H */
