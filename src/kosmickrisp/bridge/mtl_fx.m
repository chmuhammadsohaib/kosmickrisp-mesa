/*
 * SPDX-License-Identifier: MIT
 */

#include "mtl_fx.h"

#include <Metal/MTL4ArgumentTable.h>
#include <Metal/MTL4CommandBuffer.h>
#include <Metal/MTL4ComputeCommandEncoder.h>
#include <Metal/MTL4Compiler.h>
#include <Metal/MTLDevice.h>
#include <Metal/MTLTexture.h>
#include <MetalFX/MetalFX.h>

bool
mtl_fx_temporal_scaler_supported(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [MTLFXTemporalScalerDescriptor supportsMetal4FX:dev];
   }
}

float
mtl_fx_temporal_scaler_max_scale(mtl_device *device)
{
   @autoreleasepool {
      return [MTLFXTemporalScalerDescriptor
         supportedInputContentMaxScaleForDevice:(id<MTLDevice>)device];
   }
}

mtl_fx_temporal_scaler *
mtl_new_fx_temporal_scaler(mtl_device *device, mtl_compiler *compiler,
                           const struct mtl_fx_temporal_scaler_desc *desc)
{
   @autoreleasepool {
      MTLFXTemporalScalerDescriptor *d =
         [[MTLFXTemporalScalerDescriptor new] autorelease];
      d.colorTextureFormat = (MTLPixelFormat)desc->color_format;
      d.depthTextureFormat = (MTLPixelFormat)desc->depth_format;
      d.motionTextureFormat = (MTLPixelFormat)desc->motion_format;
      d.outputTextureFormat = (MTLPixelFormat)desc->output_format;
      d.inputWidth = desc->input_width;
      d.inputHeight = desc->input_height;
      d.outputWidth = desc->output_width;
      d.outputHeight = desc->output_height;
      d.autoExposureEnabled = desc->auto_exposure;
      d.inputContentPropertiesEnabled = desc->input_content_properties;
      if (desc->input_content_properties) {
         d.inputContentMinScale = desc->input_content_min_scale;
         d.inputContentMaxScale = desc->input_content_max_scale;
      }
      d.outputResolutionMotionVectorsEnabled = desc->output_resolution_motion;
      d.jitteredMotionVectorsEnabled = desc->jittered_motion;
      return [d newTemporalScalerWithDevice:(id<MTLDevice>)device
                                   compiler:(id<MTL4Compiler>)compiler];
   }
}

mtl_texture *
mtl_fx_new_output_texture(mtl_device *device, mtl_fx_temporal_scaler *scaler)
{
   @autoreleasepool {
      id<MTL4FXTemporalScaler> s = (id<MTL4FXTemporalScaler>)scaler;
      MTLTextureDescriptor *d = [MTLTextureDescriptor
         texture2DDescriptorWithPixelFormat:s.outputTextureFormat
                                      width:s.outputWidth
                                     height:s.outputHeight
                                  mipmapped:NO];
      d.storageMode = MTLStorageModePrivate;
      d.usage = s.outputTextureUsage | MTLTextureUsageShaderRead;
      return [(id<MTLDevice>)device newTextureWithDescriptor:d];
   }
}

uint32_t
mtl_fx_input_usage(mtl_fx_temporal_scaler *scaler)
{
   id<MTL4FXTemporalScaler> s = (id<MTL4FXTemporalScaler>)scaler;
   return (uint32_t)(s.colorTextureUsage | s.depthTextureUsage |
                     s.motionTextureUsage);
}

void
mtl_fx_temporal_scaler_encode(mtl_fx_temporal_scaler *scaler,
                              mtl_command_buffer *cmd_buf,
                              const struct mtl_fx_temporal_scaler_frame *frame)
{
   @autoreleasepool {
      id<MTL4FXTemporalScaler> s = (id<MTL4FXTemporalScaler>)scaler;
      s.colorTexture = (id<MTLTexture>)frame->color;
      s.depthTexture = (id<MTLTexture>)frame->depth;
      s.motionTexture = (id<MTLTexture>)frame->motion;
      s.outputTexture = (id<MTLTexture>)frame->output;
      s.exposureTexture = (id<MTLTexture>)frame->exposure;
      s.inputContentWidth = frame->content_width;
      s.inputContentHeight = frame->content_height;
      s.colorContentOffsetX = frame->color_x;
      s.colorContentOffsetY = frame->color_y;
      s.depthContentOffsetX = frame->depth_x;
      s.depthContentOffsetY = frame->depth_y;
      s.motionContentOffsetX = frame->motion_x;
      s.motionContentOffsetY = frame->motion_y;
      s.jitterOffsetX = frame->jitter_x;
      s.jitterOffsetY = frame->jitter_y;
      s.motionVectorScaleX = frame->motion_scale_x;
      s.motionVectorScaleY = frame->motion_scale_y;
      s.preExposure = frame->pre_exposure;
      s.reset = frame->reset;
      s.depthReversed = frame->depth_reversed;
      [s encodeToCommandBuffer:(id<MTL4CommandBuffer>)cmd_buf];
      /* Don't keep the app's textures alive through the scaler */
      s.colorTexture = nil;
      s.depthTexture = nil;
      s.motionTexture = nil;
      s.outputTexture = nil;
      s.exposureTexture = nil;
   }
}

bool
mtl_fx_frame_interpolator_supported(mtl_device *device)
{
   @autoreleasepool {
      return [MTLFXFrameInterpolatorDescriptor
         supportsMetal4FX:(id<MTLDevice>)device];
   }
}

mtl_fx_frame_interpolator *
mtl_new_fx_frame_interpolator(mtl_device *device, mtl_compiler *compiler,
                              const struct mtl_fx_frame_interpolator_desc *desc)
{
   @autoreleasepool {
      MTLFXFrameInterpolatorDescriptor *d =
         [[MTLFXFrameInterpolatorDescriptor new] autorelease];
      d.colorTextureFormat = (MTLPixelFormat)desc->color_format;
      d.depthTextureFormat = (MTLPixelFormat)desc->depth_format;
      d.motionTextureFormat = (MTLPixelFormat)desc->motion_format;
      d.outputTextureFormat = (MTLPixelFormat)desc->output_format;
      if (desc->ui_format != MTL_PIXEL_FORMAT_INVALID)
         d.uiTextureFormat = (MTLPixelFormat)desc->ui_format;
      d.inputWidth = desc->input_width;
      d.inputHeight = desc->input_height;
      d.outputWidth = desc->output_width;
      d.outputHeight = desc->output_height;
      /* The previous frame's color is always passed in */
      if ([d respondsToSelector:@selector(setRequiresPrevColorTexture:)])
         d.requiresPrevColorTexture = YES;
      return [d newFrameInterpolatorWithDevice:(id<MTLDevice>)device
                                      compiler:(id<MTL4Compiler>)compiler];
   }
}

static mtl_texture *
mtl_fx_new_private_texture(mtl_device *device, MTLPixelFormat format,
                           NSUInteger width, NSUInteger height,
                           MTLTextureUsage usage)
{
   MTLTextureDescriptor *d =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
   d.storageMode = MTLStorageModePrivate;
   d.usage = usage;
   return [(id<MTLDevice>)device newTextureWithDescriptor:d];
}

mtl_texture *
mtl_fx_interpolator_new_output_texture(mtl_device *device,
                                       mtl_fx_frame_interpolator *interpolator)
{
   @autoreleasepool {
      id<MTL4FXFrameInterpolator> fi = (id<MTL4FXFrameInterpolator>)interpolator;
      return mtl_fx_new_private_texture(
         device, fi.outputTextureFormat, fi.outputWidth, fi.outputHeight,
         fi.outputTextureUsage | MTLTextureUsageShaderRead);
   }
}

mtl_texture *
mtl_fx_interpolator_new_color_texture(mtl_device *device,
                                      mtl_fx_frame_interpolator *interpolator)
{
   @autoreleasepool {
      id<MTL4FXFrameInterpolator> fi = (id<MTL4FXFrameInterpolator>)interpolator;
      return mtl_fx_new_private_texture(
         device, fi.colorTextureFormat, fi.outputWidth, fi.outputHeight,
         fi.colorTextureUsage | MTLTextureUsageShaderRead);
   }
}

uint32_t
mtl_fx_interpolator_input_usage(mtl_fx_frame_interpolator *interpolator)
{
   id<MTL4FXFrameInterpolator> fi = (id<MTL4FXFrameInterpolator>)interpolator;
   return (uint32_t)(fi.colorTextureUsage | fi.depthTextureUsage |
                     fi.motionTextureUsage);
}

void
mtl_fx_frame_interpolator_encode(
   mtl_fx_frame_interpolator *interpolator, mtl_command_buffer *cmd_buf,
   const struct mtl_fx_frame_interpolator_frame *frame)
{
   @autoreleasepool {
      id<MTL4FXFrameInterpolator> fi = (id<MTL4FXFrameInterpolator>)interpolator;
      fi.colorTexture = (id<MTLTexture>)frame->color;
      fi.prevColorTexture = (id<MTLTexture>)frame->prev_color;
      fi.depthTexture = (id<MTLTexture>)frame->depth;
      fi.motionTexture = (id<MTLTexture>)frame->motion;
      fi.uiTexture = (id<MTLTexture>)frame->ui;
      fi.outputTexture = (id<MTLTexture>)frame->output;
      fi.motionVectorScaleX = frame->motion_scale_x;
      fi.motionVectorScaleY = frame->motion_scale_y;
      fi.jitterOffsetX = frame->jitter_x;
      fi.jitterOffsetY = frame->jitter_y;
      fi.deltaTime = frame->delta_time;
      fi.nearPlane = frame->near_plane;
      fi.farPlane = frame->far_plane;
      fi.fieldOfView = frame->field_of_view;
      fi.aspectRatio = frame->aspect_ratio;
      fi.shouldResetHistory = frame->reset;
      fi.depthReversed = frame->depth_reversed;
      fi.uiTextureComposited = frame->ui_composited;
      [fi encodeToCommandBuffer:(id<MTL4CommandBuffer>)cmd_buf];
      /* Don't keep the app's textures alive through the interpolator */
      fi.colorTexture = nil;
      fi.prevColorTexture = nil;
      fi.depthTexture = nil;
      fi.motionTexture = nil;
      fi.uiTexture = nil;
      fi.outputTexture = nil;
   }
}

mtl_argument_table *
mtl_fx_new_texture_table(mtl_device *device)
{
   @autoreleasepool {
      MTL4ArgumentTableDescriptor *d =
         [[MTL4ArgumentTableDescriptor new] autorelease];
      d.maxTextureBindCount = 2;
      return [(id<MTLDevice>)device newArgumentTableWithDescriptor:d
                                                             error:NULL];
   }
}

void
mtl_fx_encode_texture_pass(mtl_command_buffer *cmd_buf,
                           mtl_compute_pipeline_state *pipeline,
                           mtl_argument_table *table, mtl_texture *src,
                           mtl_texture *dst, uint32_t width, uint32_t height)
{
   @autoreleasepool {
      id<MTL4ArgumentTable> t = (id<MTL4ArgumentTable>)table;
      id<MTLComputePipelineState> p = (id<MTLComputePipelineState>)pipeline;
      [t setTexture:((id<MTLTexture>)src).gpuResourceID atIndex:0];
      [t setTexture:((id<MTLTexture>)dst).gpuResourceID atIndex:1];
      id<MTL4ComputeCommandEncoder> enc =
         [(id<MTL4CommandBuffer>)cmd_buf computeCommandEncoder];
      [enc barrierAfterQueueStages:MTLStageAll
                      beforeStages:MTLStageAll
                 visibilityOptions:MTL4VisibilityOptionNone];
      [enc setComputePipelineState:p];
      [enc setArgumentTable:t];
      const NSUInteger w = p.threadExecutionWidth;
      const NSUInteger h = p.maxTotalThreadsPerThreadgroup / w;
      [enc dispatchThreads:MTLSizeMake(width, height, 1)
         threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
      [enc barrierAfterStages:MTLStageAll
            beforeQueueStages:MTLStageAll
            visibilityOptions:MTL4VisibilityOptionNone];
      [enc endEncoding];
   }
}

enum mtl_pixel_format
mtl_texture_get_pixel_format(mtl_texture *texture)
{
   return (enum mtl_pixel_format)((id<MTLTexture>)texture).pixelFormat;
}

uint32_t
mtl_texture_get_width(mtl_texture *texture)
{
   return (uint32_t)((id<MTLTexture>)texture).width;
}

uint32_t
mtl_texture_get_height(mtl_texture *texture)
{
   return (uint32_t)((id<MTLTexture>)texture).height;
}

uint32_t
mtl_texture_get_usage(mtl_texture *texture)
{
   return (uint32_t)((id<MTLTexture>)texture).usage;
}
