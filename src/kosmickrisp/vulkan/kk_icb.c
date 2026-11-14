/*
 * SPDX-License-Identifier: MIT
 */
#include "kk_icb.h"

#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

/* One vertex per possible draw. Vertex 0 writes the execution range (the
 * app's count, clamped, or 0 when conditional rendering skips the call); each
 * vertex below the count writes its draw, or resets the slot for an empty one
 * so a slot left from earlier use never runs. Indexed draws are clamped to the
 * bound index range, as Metal does for the old path's draws (robustness). */
static const char kk_icb_encode_msl[] =
   "#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "struct ICBContainer { command_buffer cmds; };\n"
   "struct Encode {\n"
   "   ulong icb, args, count, cond, range, index_buffer;\n"
   "   uint stride, max_draws, location, mode, prim, cond_inverted, index_count;\n"
   "};\n"
   "vertex void kk_icb_encode(uint vid [[vertex_id]], constant Encode &p [[buffer(0)]])\n"
   "{\n"
   "   uint n = min(*(device const uint *)p.count, p.max_draws);\n"
   "   if (p.cond != 0ul && ((*(device const uint *)p.cond != 0u) == (p.cond_inverted != 0u)))\n"
   "      n = 0u;\n"
   "   if (vid == 0u) {\n"
   "      device MTLIndirectCommandBufferExecutionRange *r =\n"
   "         (device MTLIndirectCommandBufferExecutionRange *)p.range;\n"
   "      r->location = p.location;\n"
   "      r->length = n;\n"
   "   }\n"
   "   if (vid >= n)\n"
   "      return;\n"
   "   render_command c(((device ICBContainer *)p.icb)->cmds, p.location + vid);\n"
   "   device const uint *a = (device const uint *)(p.args + ulong(vid) * p.stride);\n"
   "   if (a[0] == 0u || a[1] == 0u) {\n"
   "      c.reset();\n"
   "      return;\n"
   "   }\n"
   "   primitive_type prim = primitive_type(p.prim);\n"
   "   if (p.mode == 0u) {\n"
   "      c.draw_primitives(prim, a[2], a[0], a[1], a[3]);\n"
   "      return;\n"
   "   }\n"
   "   if (a[2] >= p.index_count) {\n"
   "      c.reset();\n"
   "      return;\n"
   "   }\n"
   "   uint indices = min(a[0], p.index_count - a[2]);\n"
   "   if (p.mode == 1u)\n"
   "      c.draw_indexed_primitives(prim, indices, (device const ushort *)p.index_buffer + a[2], a[1], a[3], a[4]);\n"
   "   else\n"
   "      c.draw_indexed_primitives(prim, indices, (device const uint *)p.index_buffer + a[2], a[1], a[3], a[4]);\n"
   "}\n";

void
kk_icb_device_init(struct kk_device *dev)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   simple_mtx_init(&dev->icb.lock, mtx_plain);
   util_dynarray_init(&dev->icb.free_chunks, NULL);
   if (!pdev->settings.icb_draws)
      return;

   mtl_library *lib = mtl_new_library(
      dev->mtl_compiler_handle, kk_icb_encode_msl, pdev->info.msl_version,
      MTL_MATH_MODE_FAST, MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
   if (!lib)
      return;
   mtl_function_descriptor *fn =
      mtl_new_library_function_descriptor(lib, "kk_icb_encode");
   mtl_render_pipeline_descriptor *desc = mtl_new_render_pipeline_descriptor();
   mtl_render_pipeline_descriptor_set_vertex_shader(desc, fn);
   mtl_render_pipeline_descriptor_set_rasterization_enabled(desc, false);
   dev->icb.encode_pipeline =
      mtl_new_render_pipeline(dev->mtl_compiler_handle, desc);
   mtl_release(desc);
   mtl_release(fn);
   mtl_release(lib);
}

void
kk_icb_device_finish(struct kk_device *dev)
{
   util_dynarray_foreach(&dev->icb.free_chunks, struct kk_icb_chunk, c) {
      kk_device_remove_allocation_from_residency_set(dev, c->handle);
      mtl_release(c->handle);
   }
   util_dynarray_fini(&dev->icb.free_chunks);
   if (dev->icb.encode_pipeline)
      mtl_release(dev->icb.encode_pipeline);
   simple_mtx_destroy(&dev->icb.lock);
}

/* A free chunk with at least `slots`, or a new one */
static bool
kk_icb_get_chunk(struct kk_device *dev, uint32_t slots,
                 struct kk_icb_chunk *out)
{
   simple_mtx_lock(&dev->icb.lock);
   unsigned n = util_dynarray_num_elements(&dev->icb.free_chunks,
                                           struct kk_icb_chunk);
   struct kk_icb_chunk *chunks = dev->icb.free_chunks.data;
   for (unsigned i = 0; i < n; i++) {
      if (chunks[i].capacity >= slots) {
         *out = chunks[i];
         chunks[i] = chunks[n - 1];
         dev->icb.free_chunks.size -= sizeof(struct kk_icb_chunk);
         simple_mtx_unlock(&dev->icb.lock);
         return true;
      }
   }
   simple_mtx_unlock(&dev->icb.lock);

   uint32_t capacity = MAX2(slots, KK_ICB_CHUNK_SLOTS);
   out->handle = mtl_new_indirect_command_buffer(dev->mtl_handle, capacity,
                                                 &out->resource_id);
   if (!out->handle)
      return false;
   out->capacity = capacity;
   out->used = 0u;
   kk_device_add_allocation_to_residency_set(dev, out->handle);
   return true;
}

bool
kk_icb_reserve(struct kk_cmd_buffer *cmd, uint32_t count,
               struct kk_icb_chunk *chunk, uint32_t *location)
{
   struct kk_icb_cmd *icb = &cmd->icb;
   if (count > KK_ICB_PASS_SLOTS - icb->pass_slots)
      return false;

   unsigned n = util_dynarray_num_elements(&icb->chunks, struct kk_icb_chunk);
   for (; icb->chunk < n; icb->chunk++, icb->next_slot = 0u) {
      struct kk_icb_chunk *c =
         util_dynarray_element(&icb->chunks, struct kk_icb_chunk, icb->chunk);
      if (c->capacity - icb->next_slot >= count) {
         *chunk = *c;
         break;
      }
   }
   if (icb->chunk == n) {
      if (!kk_icb_get_chunk(kk_cmd_buffer_device(cmd), count, chunk))
         return false;
      util_dynarray_append(&icb->chunks, *chunk);
   }

   *location = icb->next_slot;
   icb->next_slot += count;
   icb->pass_slots += count;
   struct kk_icb_chunk *c =
      util_dynarray_element(&icb->chunks, struct kk_icb_chunk, icb->chunk);
   c->used = MAX2(c->used, icb->next_slot);
   return true;
}

void
kk_icb_cmd_release(struct kk_device *dev, struct kk_icb_cmd *icb)
{
   if (!util_dynarray_num_elements(&icb->chunks, struct kk_icb_chunk))
      return;
   util_dynarray_foreach(&icb->chunks, struct kk_icb_chunk, c) {
      if (c->used)
         mtl_indirect_command_buffer_reset(c->handle, 0u, c->used);
      c->used = 0u;
   }
   simple_mtx_lock(&dev->icb.lock);
   util_dynarray_append_dynarray(&dev->icb.free_chunks, &icb->chunks);
   simple_mtx_unlock(&dev->icb.lock);
   util_dynarray_clear(&icb->chunks);
   kk_icb_cmd_new_pass(icb);
}
