/*
 * SPDX-License-Identifier: MIT
 */

/* MetalFX temporal upscaling for the private NVX channel (kk_mfx_channel.h).
 *
 * An upscale ends the open Metal encoder, encodes an MTL4FXTemporalScaler into
 * the command buffer and then writes its output into the app's image: a copy,
 * or a contrast-adaptive sharpening pass when the app asks for sharpening.
 * MetalFX only writes private textures and KosmicKrisp images live in shared
 * memory, so every scaler owns a private output texture, which is added to
 * the device residency set. Scalers are cached per render size, output size
 * and formats; when the render size changes (dynamic resolution) the scaler
 * switched to starts without history. Scalers and sharpening pipelines are
 * kept until the device is destroyed.
 */

#include "kk_mfx.h"

#include "util/os_time.h"

#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_image.h"
#include "kk_image_view.h"
#include "kk_mfx_channel.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_compiler.h"
#include "kosmickrisp/bridge/mtl_encoder.h"
#include "kosmickrisp/bridge/mtl_fx.h"

#include "util/u_debug.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* Sharpening strength steps; a pipeline is compiled per step */
#define KK_MFX_SHARPEN_STEPS 20

struct kk_mfx_scaler {
   struct list_head link;
   struct mtl_fx_temporal_scaler_desc desc;
   mtl_fx_temporal_scaler *handle;
   mtl_texture *output;
   /* Once retired: the upscale count after which nothing in flight uses it,
    * and when it was retired (kk_mfx_releasable) */
   uint64_t release_after;
   int64_t retired_ns;
   /* The upscale count when it was last used */
   uint64_t last_used;
};

/* Every render size gets a scaler, about 300 MB at a 4K output, so a game that
 * changes its render size often would grow without bound. Keep the current
 * and previous ones; a retired scaler is released once this many more upscales
 * have been recorded, after any frame that used it (games keep at most a few
 * frames in flight), and the previous one also goes once it has been idle for
 * KK_MFX_IDLE_UPSCALES. Continuous dynamic resolution makes a new scaler per
 * size whatever the cache size, so a bigger cache only held more memory. */
#define KK_MFX_MAX_SCALERS 2
#define KK_MFX_RELEASE_DELAY 16
#define KK_MFX_IDLE_UPSCALES 900
/* A count alone isn't enough: a game can upscale or interpolate several times a
 * frame (two contexts, split views), so a retired object also waits this long,
 * far more than any frame in flight takes */
#define KK_MFX_RELEASE_NS (500 * 1000 * 1000)

static bool
kk_mfx_releasable(uint64_t release_after, int64_t retired_ns, uint64_t count)
{
   return release_after <= count &&
          os_time_get_nano() - retired_ns >= KK_MFX_RELEASE_NS;
}

static void
kk_mfx_release_scaler(struct kk_device *dev, struct kk_mfx_scaler *scaler)
{
   kk_device_remove_heap_from_residency_set(dev, scaler->output);
   mtl_release(scaler->output);
   mtl_release(scaler->handle);
   free(scaler);
}

/* DLSS -> MetalFX factors for jitter and motion vectors. Both use render
 * pixels with +y down, and MetalFX takes both as DLSS defines them: motion
 * vectors are previous minus current position, jitter is the offset applied
 * to the projection. Measured with tests/mfxUpscaleTest sweep and confirmed
 * in DOOM Eternal; KK_MFX_SIGNS=jx,jy,mx,my overrides them for experiments. */
struct kk_mfx_signs {
   float jitter_x, jitter_y, motion_x, motion_y;
};
static const struct kk_mfx_signs kk_mfx_default_signs = {1.0f, 1.0f, 1.0f, 1.0f};
static struct kk_mfx_signs kk_mfx_signs;

/* Experiment: KK_MFX_CONTENT_MODE=1 sizes one scaler for the largest render
 * size and uses MetalFX's input content properties for dynamic resolution;
 * KK_MFX_CONTENT_SCALES=min,max sets its output/input ratio range (default:
 * output over the largest render size, and the device's maximum). */
static bool kk_mfx_content_mode;
static float kk_mfx_content_scales[2];

/* Contrast-adaptive sharpening, after AMD's CAS idea: a 5-tap cross whose
 * negative lobes shrink where the neighbourhood already has high contrast or
 * is near clipping. The amount is computed on x / (1 + x) so HDR highlights
 * don't sharpen to ringing. SHARPNESS (0..1) is baked in per pipeline. */
static const char kk_mfx_sharpen_source[] =
   "#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "static float3 kk_texel(texture2d<float, access::read> t, int2 p, int2 hi)\n"
   "{\n"
   "   return t.read(uint2(clamp(p, int2(0), hi))).rgb;\n"
   "}\n"
   "kernel void kk_mfx_sharpen(texture2d<float, access::read> src [[texture(0)]],\n"
   "                           texture2d<float, access::write> dst [[texture(1)]],\n"
   "                           uint2 gid [[thread_position_in_grid]])\n"
   "{\n"
   "   const int2 hi = int2(src.get_width(), src.get_height()) - 1;\n"
   "   if (gid.x > uint(hi.x) || gid.y > uint(hi.y))\n"
   "      return;\n"
   "   const int2 p = int2(gid);\n"
   "   const float3 n = kk_texel(src, p + int2(0, -1), hi);\n"
   "   const float3 w = kk_texel(src, p + int2(-1, 0), hi);\n"
   "   const float4 c4 = src.read(gid);\n"
   "   const float3 c = c4.rgb;\n"
   "   const float3 e = kk_texel(src, p + int2(1, 0), hi);\n"
   "   const float3 s = kk_texel(src, p + int2(0, 1), hi);\n"
   "   const float3 tn = n / (1.0 + n), tw = w / (1.0 + w), tc = c / (1.0 + c);\n"
   "   const float3 te = e / (1.0 + e), ts = s / (1.0 + s);\n"
   "   const float3 lo = min(min(min(tn, tw), min(te, ts)), tc);\n"
   "   const float3 up = max(max(max(tn, tw), max(te, ts)), tc);\n"
   "   const float3 amount = sqrt(saturate(min(lo, 1.0 - up) / max(up, 1e-5)));\n"
   "   const float3 lobe = amount * (-1.0 / mix(8.0, 5.0, SHARPNESS));\n"
   "   const float3 result = (lobe * (n + w + e + s) + c) / (1.0 + 4.0 * lobe);\n"
   "   dst.write(float4(max(result, 0.0), c4.a), uint2(OUTPUT_X, OUTPUT_Y) + gid);\n"
   "}\n";

/* Frame generation keeps one interpolator; it holds a private output and a
 * copy of the last color, which the next frame interpolates from. */
struct kk_mfx_interpolator {
   struct list_head link;
   struct mtl_fx_frame_interpolator_desc desc;
   mtl_fx_frame_interpolator *handle;
   mtl_texture *output;
   mtl_texture *prev_color;
   bool has_prev_color;
   /* Once retired: the interpolation count after which nothing in flight
    * uses it, and when it was retired (kk_mfx_releasable) */
   uint64_t release_after;
   int64_t retired_ns;
};

static uint64_t kk_mfx_interpolate_count;

static void
kk_mfx_release_interpolator(struct kk_device *dev,
                            struct kk_mfx_interpolator *interpolator)
{
   kk_device_remove_heap_from_residency_set(dev, interpolator->output);
   kk_device_remove_heap_from_residency_set(dev, interpolator->prev_color);
   mtl_release(interpolator->output);
   mtl_release(interpolator->prev_color);
   mtl_release(interpolator->handle);
   free(interpolator);
}

struct kk_mfx_sharpener {
   struct list_head link;
   uint32_t step, output_x, output_y;
   mtl_compute_pipeline_state *pipeline;
};

void
kk_mfx_init(struct kk_device *dev)
{
   simple_mtx_init(&dev->mfx.mutex, mtx_plain);
   list_inithead(&dev->mfx.scalers);
   list_inithead(&dev->mfx.sharpeners);
   list_inithead(&dev->mfx.retired);
   dev->mfx.scaler_count = 0;
   dev->mfx.created_count = 0;
   dev->mfx.upscale_count = 0;
   dev->mfx.texture_table = NULL;
   dev->mfx.last_scaler = NULL;
   dev->mfx.last_output = NULL;
   dev->mfx.interpolator = NULL;
   list_inithead(&dev->mfx.retired_interpolators);

   kk_mfx_signs = kk_mfx_default_signs;
   kk_mfx_content_mode = debug_get_bool_option("KK_MFX_CONTENT_MODE", false);
   kk_mfx_content_scales[0] = kk_mfx_content_scales[1] = 0.0f;
   const char *content_scales = getenv("KK_MFX_CONTENT_SCALES");
   if (content_scales)
      sscanf(content_scales, "%f,%f", &kk_mfx_content_scales[0],
             &kk_mfx_content_scales[1]);
   const char *signs = getenv("KK_MFX_SIGNS");
   if (signs)
      sscanf(signs, "%f,%f,%f,%f", &kk_mfx_signs.jitter_x,
             &kk_mfx_signs.jitter_y, &kk_mfx_signs.motion_x,
             &kk_mfx_signs.motion_y);
}

/* Releases the scalers and interpolators (a few hundred MB at 4K) once the last
 * channel module goes: kk-ffx and kk-ngx destroy theirs when a game turns FSR or
 * DLSS off or recreates its contexts. Nothing recorded may use them any more, so
 * the device is waited on first; this is rare. */
void
kk_mfx_module_created(struct kk_device *dev)
{
   simple_mtx_lock(&dev->mfx.mutex);
   dev->mfx.modules++;
   simple_mtx_unlock(&dev->mfx.mutex);
}

void
kk_mfx_module_destroyed(struct kk_device *dev)
{
   simple_mtx_lock(&dev->mfx.mutex);
   const bool last = --dev->mfx.modules == 0;
   simple_mtx_unlock(&dev->mfx.mutex);
   if (!last)
      return;

   dev->vk.dispatch_table.DeviceWaitIdle(kk_device_to_handle(dev));

   simple_mtx_lock(&dev->mfx.mutex);
   if (dev->mfx.modules == 0) {
      list_for_each_entry_safe(struct kk_mfx_scaler, scaler,
                               &dev->mfx.scalers, link)
         kk_mfx_release_scaler(dev, scaler);
      list_for_each_entry_safe(struct kk_mfx_scaler, scaler,
                               &dev->mfx.retired, link)
         kk_mfx_release_scaler(dev, scaler);
      list_inithead(&dev->mfx.scalers);
      list_inithead(&dev->mfx.retired);
      dev->mfx.scaler_count = 0;
      dev->mfx.last_scaler = NULL;
      dev->mfx.last_output = NULL;
      if (dev->mfx.interpolator)
         kk_mfx_release_interpolator(dev, dev->mfx.interpolator);
      dev->mfx.interpolator = NULL;
      list_for_each_entry_safe(struct kk_mfx_interpolator, interpolator,
                               &dev->mfx.retired_interpolators, link)
         kk_mfx_release_interpolator(dev, interpolator);
      list_inithead(&dev->mfx.retired_interpolators);
   }
   simple_mtx_unlock(&dev->mfx.mutex);
}

void
kk_mfx_finish(struct kk_device *dev)
{
   list_for_each_entry_safe(struct kk_mfx_scaler, scaler, &dev->mfx.scalers,
                            link)
      kk_mfx_release_scaler(dev, scaler);
   list_for_each_entry_safe(struct kk_mfx_scaler, scaler, &dev->mfx.retired,
                            link)
      kk_mfx_release_scaler(dev, scaler);
   list_for_each_entry_safe(struct kk_mfx_sharpener, sharpener,
                            &dev->mfx.sharpeners, link) {
      mtl_release(sharpener->pipeline);
      free(sharpener);
   }
   if (dev->mfx.interpolator)
      kk_mfx_release_interpolator(dev, dev->mfx.interpolator);
   list_for_each_entry_safe(struct kk_mfx_interpolator, interpolator,
                            &dev->mfx.retired_interpolators, link)
      kk_mfx_release_interpolator(dev, interpolator);
   if (dev->mfx.texture_table)
      mtl_release(dev->mfx.texture_table);
   simple_mtx_destroy(&dev->mfx.mutex);
}

struct kk_image_view *
kk_mfx_view_from_handle(struct kk_device *dev, uint64_t handle)
{
   struct kk_image_view *view = (struct kk_image_view *)(uintptr_t)handle;
   if (!view || view->vk.base.type != VK_OBJECT_TYPE_IMAGE_VIEW ||
       view->vk.base.device != &dev->vk)
      return NULL;
   return view;
}

static struct kk_mfx_scaler *
kk_mfx_get_scaler(struct kk_device *dev,
                  const struct mtl_fx_temporal_scaler_desc *desc)
{
   simple_mtx_lock(&dev->mfx.mutex);
   const uint64_t upscale = ++dev->mfx.upscale_count;
   list_for_each_entry_safe(struct kk_mfx_scaler, retired, &dev->mfx.retired,
                            link) {
      if (kk_mfx_releasable(retired->release_after, retired->retired_ns,
                            upscale)) {
         list_del(&retired->link);
         kk_mfx_release_scaler(dev, retired);
      }
   }
   list_for_each_entry_safe(struct kk_mfx_scaler, scaler, &dev->mfx.scalers,
                            link) {
      if (memcmp(&scaler->desc, desc, sizeof(*desc)) == 0) {
         /* Most recently used first */
         list_del(&scaler->link);
         list_add(&scaler->link, &dev->mfx.scalers);
         scaler->last_used = upscale;
         simple_mtx_unlock(&dev->mfx.mutex);
         return scaler;
      }
      /* Idle for a while, or for another output: the output resolution or
       * format changed, and a scaler (hundreds of MB at 4K) for the old one
       * is only wanted again if it changes back */
      const bool other_output = scaler->desc.output_width != desc->output_width ||
                                scaler->desc.output_height != desc->output_height ||
                                scaler->desc.output_format != desc->output_format;
      if (scaler->last_used + KK_MFX_IDLE_UPSCALES < upscale || other_output) {
         list_del(&scaler->link);
         scaler->release_after = upscale + KK_MFX_RELEASE_DELAY;
         scaler->retired_ns = os_time_get_nano();
         list_add(&scaler->link, &dev->mfx.retired);
         dev->mfx.scaler_count--;
         if (dev->mfx.last_scaler == scaler)
            dev->mfx.last_scaler = NULL;
      }
   }

   struct kk_mfx_scaler *scaler = calloc(1, sizeof(*scaler));
   if (scaler) {
      scaler->desc = *desc;
      scaler->handle = mtl_new_fx_temporal_scaler(
         dev->mtl_handle, dev->mtl_compiler_handle, desc);
      if (scaler->handle)
         scaler->output = mtl_fx_new_output_texture(dev->mtl_handle,
                                                    scaler->handle);
      if (!scaler->output) {
         if (scaler->handle)
            mtl_release(scaler->handle);
         free(scaler);
         scaler = NULL;
      }
   }
   if (scaler) {
      kk_device_add_heap_to_residency_set(dev, scaler->output);
      scaler->last_used = upscale;
      list_add(&scaler->link, &dev->mfx.scalers);
      if (++dev->mfx.scaler_count > KK_MFX_MAX_SCALERS) {
         struct kk_mfx_scaler *oldest =
            list_last_entry(&dev->mfx.scalers, struct kk_mfx_scaler, link);
         list_del(&oldest->link);
         oldest->release_after = upscale + KK_MFX_RELEASE_DELAY;
         oldest->retired_ns = os_time_get_nano();
         list_add(&oldest->link, &dev->mfx.retired);
         dev->mfx.scaler_count--;
         if (dev->mfx.last_scaler == oldest)
            dev->mfx.last_scaler = NULL;
      }
   }
   const uint32_t created = ++dev->mfx.created_count;
   simple_mtx_unlock(&dev->mfx.mutex);

   /* Every new render size logs; with frequent changes only some do */
   if (scaler && created > 32 && created % 100 != 0)
      return scaler;
   fprintf(stderr,
           "[kk] mfx: %s temporal scaler #%u %ux%u -> %ux%u (formats color %u "
           "depth %u motion %u output %u, auto exposure %d, output-res motion "
           "%d)\n",
           scaler ? "created" : "FAILED to create", created, desc->input_width,
           desc->input_height, desc->output_width, desc->output_height,
           desc->color_format, desc->depth_format, desc->motion_format,
           desc->output_format, desc->auto_exposure,
           desc->output_resolution_motion);
   return scaler;
}

/* The sharpening pipeline for a strength step and output offset */
static mtl_compute_pipeline_state *
kk_mfx_get_sharpener(struct kk_device *dev, uint32_t step, uint32_t output_x,
                     uint32_t output_y)
{
   mtl_compute_pipeline_state *pipeline = NULL;

   simple_mtx_lock(&dev->mfx.mutex);
   list_for_each_entry(struct kk_mfx_sharpener, sharpener,
                       &dev->mfx.sharpeners, link) {
      if (sharpener->step == step && sharpener->output_x == output_x &&
          sharpener->output_y == output_y) {
         pipeline = sharpener->pipeline;
         break;
      }
   }
   if (!pipeline) {
      char defines[160];
      snprintf(defines, sizeof(defines),
               "#define SHARPNESS %.4f\n#define OUTPUT_X %uu\n"
               "#define OUTPUT_Y %uu\n",
               (float)step / KK_MFX_SHARPEN_STEPS, output_x, output_y);
      char *source = malloc(strlen(defines) + sizeof(kk_mfx_sharpen_source));
      mtl_library *library = NULL;
      mtl_function_descriptor *function = NULL;
      if (source) {
         strcpy(source, defines);
         strcat(source, kk_mfx_sharpen_source);
         library = mtl_new_library(dev->mtl_compiler_handle, source,
                                   MTL_LANGUAGE_VERSION_4_0,
                                   MTL_MATH_MODE_FAST,
                                   MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
         free(source);
      }
      if (library)
         function =
            mtl_new_library_function_descriptor(library, "kk_mfx_sharpen");
      if (function)
         pipeline =
            mtl_new_compute_pipeline_state(dev->mtl_compiler_handle, function, 256);
      struct kk_mfx_sharpener *sharpener =
         pipeline ? calloc(1, sizeof(*sharpener)) : NULL;
      if (sharpener) {
         sharpener->step = step;
         sharpener->output_x = output_x;
         sharpener->output_y = output_y;
         sharpener->pipeline = pipeline;
         list_add(&sharpener->link, &dev->mfx.sharpeners);
      } else if (pipeline) {
         mtl_release(pipeline);
         pipeline = NULL;
      }
      if (function)
         mtl_release(function);
      if (library)
         mtl_release(library);
      if (!dev->mfx.texture_table)
         dev->mfx.texture_table = mtl_fx_new_texture_table(dev->mtl_handle);
      if (!pipeline || !dev->mfx.texture_table)
         fprintf(stderr, "[kk] mfx: couldn't build the sharpening pass\n");
   }
   simple_mtx_unlock(&dev->mfx.mutex);
   return dev->mfx.texture_table ? pipeline : NULL;
}

static mtl_texture *
kk_mfx_view_texture(const struct kk_image_view *view)
{
   return view->planes[0].mtl_handle_sampled;
}

static void
kk_mfx_warn(const char *message, uint64_t handle)
{
   static uint32_t warn_count;
   if (p_atomic_inc_return(&warn_count) <= 16)
      fprintf(stderr, "[kk] mfx: skipped: %s (0x%" PRIx64 ")\n",
              message, handle);
}

void
kk_mfx_cmd_upscale(struct kk_cmd_buffer *cmd,
                   const struct kk_mfx_upscale_params *params)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   if (params->magic != KK_MFX_PARAMS_MAGIC ||
       params->version != KK_MFX_PARAMS_VERSION) {
      kk_mfx_warn("bad parameter block", params->magic);
      return;
   }

   const struct {
      uint64_t handle;
      const char *name;
   } inputs[] = {
      {params->color, "color"},
      {params->depth, "depth"},
      {params->motion, "motion"},
      {params->output, "output"},
   };
   struct kk_image_view *views[4];
   for (unsigned i = 0; i < ARRAY_SIZE(inputs); i++) {
      views[i] = kk_mfx_view_from_handle(dev, inputs[i].handle);
      if (!views[i]) {
         kk_mfx_warn(inputs[i].name, inputs[i].handle);
         return;
      }
   }
   const bool auto_exposure = params->flags & KK_MFX_FLAG_AUTO_EXPOSURE;
   struct kk_image_view *exposure_view =
      (!auto_exposure && params->exposure)
         ? kk_mfx_view_from_handle(dev, params->exposure)
         : NULL;

   mtl_texture *color = kk_mfx_view_texture(views[0]);
   mtl_texture *depth = kk_mfx_view_texture(views[1]);
   mtl_texture *motion = kk_mfx_view_texture(views[2]);
   struct kk_image_view *output_view = views[3];
   struct kk_image *output_image =
      container_of(output_view->vk.image, struct kk_image, vk);
   mtl_texture *output = output_image->planes[0].mtl_handle;

   if (!params->render_width || !params->render_height ||
       !params->output_width || !params->output_height) {
      kk_mfx_warn("empty render or output size", 0);
      return;
   }
   /* MetalFX aborts the process beyond its largest ratio (3x per axis) */
   const float limit = mtl_fx_temporal_scaler_max_scale(dev->mtl_handle);
   if (params->output_width > limit * params->render_width ||
       params->output_height > limit * params->render_height) {
      kk_mfx_warn("render size below MetalFX's limit (output / 3)",
                  ((uint64_t)params->render_width << 32) | params->render_height);
      return;
   }
   const bool content_mode = kk_mfx_content_mode;
   const uint32_t input_width =
      content_mode ? MAX2(params->max_render_width, params->render_width)
                   : params->render_width;
   const uint32_t input_height =
      content_mode ? MAX2(params->max_render_height, params->render_height)
                   : params->render_height;
   float min_scale = 0.0f, max_scale = 0.0f;
   if (content_mode) {
      min_scale = kk_mfx_content_scales[0] > 0.0f
                     ? kk_mfx_content_scales[0]
                     : MAX2(1.0f, MIN2((float)params->output_width / input_width,
                                       (float)params->output_height / input_height));
      max_scale = kk_mfx_content_scales[1] > 0.0f
                     ? kk_mfx_content_scales[1]
                     : mtl_fx_temporal_scaler_max_scale(dev->mtl_handle);
      max_scale = MAX2(max_scale, min_scale);
   }
   const struct mtl_fx_temporal_scaler_desc desc = {
      .color_format = mtl_texture_get_pixel_format(color),
      .depth_format = mtl_texture_get_pixel_format(depth),
      .motion_format = mtl_texture_get_pixel_format(motion),
      .output_format = mtl_texture_get_pixel_format(output),
      .input_width = input_width,
      .input_height = input_height,
      .output_width = params->output_width,
      .output_height = params->output_height,
      .auto_exposure = auto_exposure,
      /* MetalFX's input content mode is not used: it costs too much quality
       * (mfxUpscaleTest dynamic: 0.083 vs 0.043 at a constant 192 in 256
       * textures). Dynamic resolution uses a scaler per render size instead. */
      .input_content_properties = content_mode,
      .input_content_min_scale = min_scale,
      .input_content_max_scale = max_scale,
      .output_resolution_motion = !(params->flags & KK_MFX_FLAG_MV_LOW_RES),
      .jittered_motion = params->flags & KK_MFX_FLAG_MV_JITTERED,
   };
   struct kk_mfx_scaler *scaler = kk_mfx_get_scaler(dev, &desc);
   if (!scaler) {
      kk_mfx_warn("no scaler for these formats and sizes", params->color);
      return;
   }

   const uint32_t needed = mtl_fx_input_usage(scaler->handle);
   if ((mtl_texture_get_usage(color) & needed) != needed ||
       (mtl_texture_get_usage(depth) & needed) != needed ||
       (mtl_texture_get_usage(motion) & needed) != needed) {
      kk_mfx_warn("an input lacks the usage MetalFX needs", needed);
      return;
   }

   /* A render size change for the same output switches scalers; the one
    * switched to holds history from whenever it last ran, so start it over. */
   simple_mtx_lock(&dev->mfx.mutex);
   const bool reset = (params->flags & KK_MFX_FLAG_RESET) ||
                      (dev->mfx.last_output == output &&
                       dev->mfx.last_scaler != scaler);
   simple_mtx_unlock(&dev->mfx.mutex);

   /* MetalFX encodes its own passes, so no encoder may be open. Leaving a
    * render pass is what cs_get_compute does too. */
   if (cmd->metal.render) {
      kk_apply_attachment_store_ops(cmd, true);
      cs_end(cmd);
      kk_cmd_buffer_dirty_all_gfx(cmd);
      cmd->state.gfx.need_to_start_render_pass = true;
   }
   cs_end(cmd);

   /* The previous encoder ended with a barrier before all later queue stages,
    * so MetalFX sees its writes. */
   const struct mtl_fx_temporal_scaler_frame frame = {
      .color = color,
      .depth = depth,
      .motion = motion,
      .output = scaler->output,
      .exposure = exposure_view ? kk_mfx_view_texture(exposure_view) : NULL,
      /* Within what the scaler accepts: at most its input size, at least the
       * output over its largest ratio */
      .content_width = params->render_width,
      .content_height = params->render_height,
      .color_x = params->color_x,
      .color_y = params->color_y,
      .depth_x = params->depth_x,
      .depth_y = params->depth_y,
      .motion_x = params->motion_x,
      .motion_y = params->motion_y,
      .jitter_x = kk_mfx_signs.jitter_x * params->jitter_x,
      .jitter_y = kk_mfx_signs.jitter_y * params->jitter_y,
      .motion_scale_x = kk_mfx_signs.motion_x * params->mv_scale_x,
      .motion_scale_y = kk_mfx_signs.motion_y * params->mv_scale_y,
      .pre_exposure = params->pre_exposure > 0.0f ? params->pre_exposure : 1.0f,
      .reset = reset,
      .depth_reversed = params->flags & KK_MFX_FLAG_DEPTH_INVERTED,
   };
   /* The scaler's per-frame state is shared; encode one frame at a time */
   simple_mtx_lock(&dev->mfx.mutex);
   dev->mfx.last_scaler = scaler;
   dev->mfx.last_output = output;
   mtl_fx_temporal_scaler_encode(scaler->handle, cmd->metal.cmd_buf, &frame);
   simple_mtx_unlock(&dev->mfx.mutex);

   /* Sharpen into the app's image, if it asked and can be written by shaders */
   const uint32_t sharpen_step =
      (uint32_t)roundf(CLAMP(params->sharpness, 0.0f, 1.0f) * KK_MFX_SHARPEN_STEPS);
   if (sharpen_step &&
       (mtl_texture_get_usage(output) & MTL_TEXTURE_USAGE_SHADER_WRITE) &&
       output_view->vk.base_array_layer == 0 && output_view->vk.base_mip_level == 0) {
      mtl_compute_pipeline_state *pipeline =
         kk_mfx_get_sharpener(dev, sharpen_step, params->output_x, params->output_y);
      if (pipeline) {
         simple_mtx_lock(&dev->mfx.mutex);
         mtl_fx_encode_texture_pass(cmd->metal.cmd_buf, pipeline,
                                    dev->mfx.texture_table, scaler->output,
                                    output, params->output_width,
                                    params->output_height);
         simple_mtx_unlock(&dev->mfx.mutex);
         return;
      }
   }

   /* Copy into the app's image once MetalFX is done writing */
   mtl_compute_encoder *encoder = cs_get_compute(cmd);
   mtl_barrier_after_queue_stages(encoder, MTL_STAGE_ALL, MTL_STAGE_ALL);
   mtl_copy_from_texture_to_texture(
      encoder, scaler->output, 0, 0, (struct mtl_origin){0, 0, 0},
      (struct mtl_size){params->output_width, params->output_height, 1},
      output, output_view->vk.base_array_layer, output_view->vk.base_mip_level,
      (struct mtl_origin){params->output_x, params->output_y, 0});
}

/* The interpolator for these inputs, creating it (and retiring the previous
 * one) when they change. Called with dev->mfx.mutex held. */
static struct kk_mfx_interpolator *
kk_mfx_get_interpolator(struct kk_device *dev,
                        const struct mtl_fx_frame_interpolator_desc *desc)
{
   const uint64_t count = ++kk_mfx_interpolate_count;
   list_for_each_entry_safe(struct kk_mfx_interpolator, retired,
                            &dev->mfx.retired_interpolators, link) {
      if (kk_mfx_releasable(retired->release_after, retired->retired_ns,
                            count)) {
         list_del(&retired->link);
         kk_mfx_release_interpolator(dev, retired);
      }
   }

   struct kk_mfx_interpolator *current = dev->mfx.interpolator;
   if (current && memcmp(&current->desc, desc, sizeof(*desc)) == 0)
      return current;

   struct kk_mfx_interpolator *interpolator = calloc(1, sizeof(*interpolator));
   if (!interpolator)
      return NULL;
   interpolator->desc = *desc;
   interpolator->handle = mtl_new_fx_frame_interpolator(
      dev->mtl_handle, dev->mtl_compiler_handle, desc);
   if (interpolator->handle) {
      interpolator->output = mtl_fx_interpolator_new_output_texture(
         dev->mtl_handle, interpolator->handle);
      interpolator->prev_color = mtl_fx_interpolator_new_color_texture(
         dev->mtl_handle, interpolator->handle);
   }
   if (!interpolator->output || !interpolator->prev_color) {
      if (interpolator->output)
         mtl_release(interpolator->output);
      if (interpolator->prev_color)
         mtl_release(interpolator->prev_color);
      if (interpolator->handle)
         mtl_release(interpolator->handle);
      free(interpolator);
      return NULL;
   }
   kk_device_add_heap_to_residency_set(dev, interpolator->output);
   kk_device_add_heap_to_residency_set(dev, interpolator->prev_color);

   if (current) {
      current->release_after = count + KK_MFX_RELEASE_DELAY;
      current->retired_ns = os_time_get_nano();
      list_add(&current->link, &dev->mfx.retired_interpolators);
   }
   dev->mfx.interpolator = interpolator;

   fprintf(stderr,
           "[kk] mfx: created frame interpolator %ux%u (depth/motion %ux%u, "
           "formats color %u depth %u motion %u output %u ui %u)\n",
           desc->output_width, desc->output_height, desc->input_width,
           desc->input_height, desc->color_format, desc->depth_format,
           desc->motion_format, desc->output_format, desc->ui_format);
   return interpolator;
}

void
kk_mfx_cmd_interpolate(struct kk_cmd_buffer *cmd,
                       const struct kk_mfx_interpolate_params *params)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   if (params->magic != KK_MFX_INTERPOLATE_MAGIC ||
       params->version != KK_MFX_INTERPOLATE_VERSION) {
      kk_mfx_warn("bad interpolation parameter block", params->magic);
      return;
   }
   if (!params->width || !params->height || !params->input_width ||
       !params->input_height) {
      kk_mfx_warn("empty interpolation size", 0);
      return;
   }

   const struct {
      uint64_t handle;
      const char *name;
   } inputs[] = {
      {params->color, "color"},
      {params->depth, "depth"},
      {params->motion, "motion"},
      {params->output, "output"},
   };
   struct kk_image_view *views[4];
   for (unsigned i = 0; i < ARRAY_SIZE(inputs); i++) {
      views[i] = kk_mfx_view_from_handle(dev, inputs[i].handle);
      if (!views[i]) {
         kk_mfx_warn(inputs[i].name, inputs[i].handle);
         return;
      }
   }
   struct kk_image_view *prev_view =
      params->prev_color ? kk_mfx_view_from_handle(dev, params->prev_color)
                         : NULL;
   const bool ui_separate = params->flags & KK_MFX_INTERPOLATE_UI_SEPARATE;
   struct kk_image_view *ui_view =
      params->ui ? kk_mfx_view_from_handle(dev, params->ui) : NULL;

   mtl_texture *color = kk_mfx_view_texture(views[0]);
   mtl_texture *depth = kk_mfx_view_texture(views[1]);
   mtl_texture *motion = kk_mfx_view_texture(views[2]);
   struct kk_image_view *output_view = views[3];
   struct kk_image *output_image =
      container_of(output_view->vk.image, struct kk_image, vk);
   mtl_texture *output = output_image->planes[0].mtl_handle;
   mtl_texture *ui = ui_view ? kk_mfx_view_texture(ui_view) : NULL;

   const struct mtl_fx_frame_interpolator_desc desc = {
      .color_format = mtl_texture_get_pixel_format(color),
      .depth_format = mtl_texture_get_pixel_format(depth),
      .motion_format = mtl_texture_get_pixel_format(motion),
      .output_format = mtl_texture_get_pixel_format(output),
      .ui_format = ui ? mtl_texture_get_pixel_format(ui) : MTL_PIXEL_FORMAT_INVALID,
      .input_width = params->input_width,
      .input_height = params->input_height,
      .output_width = params->width,
      .output_height = params->height,
   };

   simple_mtx_lock(&dev->mfx.mutex);
   struct kk_mfx_interpolator *interpolator = kk_mfx_get_interpolator(dev, &desc);
   simple_mtx_unlock(&dev->mfx.mutex);
   if (!interpolator) {
      kk_mfx_warn("no frame interpolator for these formats and sizes",
                  params->color);
      return;
   }
   const uint32_t needed = mtl_fx_interpolator_input_usage(interpolator->handle);
   if ((mtl_texture_get_usage(color) & needed) != needed ||
       (mtl_texture_get_usage(depth) & needed) != needed ||
       (mtl_texture_get_usage(motion) & needed) != needed) {
      kk_mfx_warn("an interpolation input lacks the usage MetalFX needs", needed);
      return;
   }

   /* MetalFX encodes its own passes, so no encoder may be open */
   if (cmd->metal.render) {
      kk_apply_attachment_store_ops(cmd, true);
      cs_end(cmd);
      kk_cmd_buffer_dirty_all_gfx(cmd);
      cmd->state.gfx.need_to_start_render_pass = true;
   }
   cs_end(cmd);

   simple_mtx_lock(&dev->mfx.mutex);
   /* Without a previous frame there is nothing to interpolate from yet: the
    * current color stands in, with history reset */
   mtl_texture *prev = prev_view ? kk_mfx_view_texture(prev_view)
                                 : interpolator->prev_color;
   const bool first = !prev_view && !interpolator->has_prev_color;
   const float fov_degrees =
      params->fov > 0.0f ? params->fov * (180.0f / (float)M_PI) : 60.0f;
   const struct mtl_fx_frame_interpolator_frame frame = {
      .color = color,
      .prev_color = first ? color : prev,
      .depth = depth,
      .motion = motion,
      .ui = ui,
      .output = interpolator->output,
      .motion_scale_x = kk_mfx_signs.motion_x * params->mv_scale_x,
      .motion_scale_y = kk_mfx_signs.motion_y * params->mv_scale_y,
      .jitter_x = kk_mfx_signs.jitter_x * params->jitter_x,
      .jitter_y = kk_mfx_signs.jitter_y * params->jitter_y,
      .delta_time = params->delta_time > 0.0f ? params->delta_time : 1.0f / 60.0f,
      .near_plane = params->near_plane > 0.0f ? params->near_plane : 0.1f,
      .far_plane = params->far_plane > 0.0f ? params->far_plane : 10000.0f,
      .field_of_view = fov_degrees,
      .aspect_ratio = params->aspect_ratio > 0.0f
                         ? params->aspect_ratio
                         : (float)params->width / params->height,
      .reset = first || (params->flags & KK_MFX_INTERPOLATE_RESET),
      .depth_reversed = params->flags & KK_MFX_INTERPOLATE_DEPTH_INVERTED,
      .ui_composited = ui && !ui_separate,
   };
   mtl_fx_frame_interpolator_encode(interpolator->handle, cmd->metal.cmd_buf,
                                    &frame);
   interpolator->has_prev_color = true;
   simple_mtx_unlock(&dev->mfx.mutex);

   /* Once MetalFX is done: the generated frame into the app's image, and this
    * frame's color kept for the next interpolation */
   mtl_compute_encoder *encoder = cs_get_compute(cmd);
   mtl_barrier_after_queue_stages(encoder, MTL_STAGE_ALL, MTL_STAGE_ALL);
   mtl_copy_from_texture_to_texture(
      encoder, interpolator->output, 0, 0, (struct mtl_origin){0, 0, 0},
      (struct mtl_size){params->width, params->height, 1}, output,
      output_view->vk.base_array_layer, output_view->vk.base_mip_level,
      (struct mtl_origin){params->output_x, params->output_y, 0});
   if (!prev_view) {
      mtl_copy_from_texture_to_texture(
         encoder, color, 0, 0,
         (struct mtl_origin){params->color_x, params->color_y, 0},
         (struct mtl_size){params->width, params->height, 1},
         interpolator->prev_color, 0, 0, (struct mtl_origin){0, 0, 0});
   }
   struct kk_image_view *real_view =
      params->output_real ? kk_mfx_view_from_handle(dev, params->output_real)
                          : NULL;
   /* Not when the real frame is already where the output goes (another view of
    * the same image): copying a texture onto itself is undefined in Metal */
   if (real_view &&
       (real_view->vk.image != views[0]->vk.image ||
        real_view->vk.base_array_layer != views[0]->vk.base_array_layer ||
        real_view->vk.base_mip_level != views[0]->vk.base_mip_level ||
        params->color_x != params->output_x ||
        params->color_y != params->output_y)) {
      struct kk_image *real_image =
         container_of(real_view->vk.image, struct kk_image, vk);
      mtl_copy_from_texture_to_texture(
         encoder, color, 0, 0,
         (struct mtl_origin){params->color_x, params->color_y, 0},
         (struct mtl_size){params->width, params->height, 1},
         real_image->planes[0].mtl_handle, real_view->vk.base_array_layer,
         real_view->vk.base_mip_level,
         (struct mtl_origin){params->output_x, params->output_y, 0});
   }
}
