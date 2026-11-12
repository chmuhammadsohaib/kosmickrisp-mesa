/*
 * SPDX-License-Identifier: MIT
 */

#include "mtl_fx.h"

bool
mtl_fx_temporal_scaler_supported(mtl_device *device)
{
   return false;
}

float
mtl_fx_temporal_scaler_max_scale(mtl_device *device)
{
   return 1.0f;
}

mtl_fx_temporal_scaler *
mtl_new_fx_temporal_scaler(mtl_device *device, mtl_compiler *compiler,
                           const struct mtl_fx_temporal_scaler_desc *desc)
{
   return NULL;
}

mtl_texture *
mtl_fx_new_output_texture(mtl_device *device, mtl_fx_temporal_scaler *scaler)
{
   return NULL;
}

uint32_t
mtl_fx_input_usage(mtl_fx_temporal_scaler *scaler)
{
   return 0u;
}

void
mtl_fx_temporal_scaler_encode(mtl_fx_temporal_scaler *scaler,
                              mtl_command_buffer *cmd_buf,
                              const struct mtl_fx_temporal_scaler_frame *frame)
{
}

mtl_argument_table *
mtl_fx_new_texture_table(mtl_device *device)
{
   return NULL;
}

void
mtl_fx_encode_texture_pass(mtl_command_buffer *cmd_buf,
                           mtl_compute_pipeline_state *pipeline,
                           mtl_argument_table *table, mtl_texture *src,
                           mtl_texture *dst, uint32_t width, uint32_t height)
{
}

enum mtl_pixel_format
mtl_texture_get_pixel_format(mtl_texture *texture)
{
   return MTL_PIXEL_FORMAT_INVALID;
}

uint32_t
mtl_texture_get_width(mtl_texture *texture)
{
   return 0u;
}

uint32_t
mtl_texture_get_height(mtl_texture *texture)
{
   return 0u;
}

uint32_t
mtl_texture_get_usage(mtl_texture *texture)
{
   return 0u;
}
