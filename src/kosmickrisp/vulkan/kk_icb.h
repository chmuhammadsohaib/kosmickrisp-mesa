/*
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_ICB_H
#define KK_ICB_H 1

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/simple_mtx.h"
#include "util/u_dynarray.h"

#include <stdbool.h>
#include <stdint.h>

struct kk_device;
struct kk_cmd_buffer;

/* KK_ICB_DRAWS (off by default): vkCmdDraw*IndirectCount as GPU-built draws.
 * A rasterization-off vertex pass in the render pass writes the draws the
 * count buffer asks for into an indirect command buffer, then the ICB runs
 * with that count as its range: no Metal draw per possible draw, no compute
 * pass that splits the render pass. */

/* Slots in a pooled ICB; calls with more possible draws get one of their own */
#define KK_ICB_CHUNK_SLOTS 4096u
/* Slots one render pass may use (about 45 MB); later calls take the old path */
#define KK_ICB_PASS_SLOTS 65536u

struct kk_icb_chunk {
   mtl_indirect_command_buffer *handle;
   uint64_t resource_id;
   uint32_t capacity;
};

struct kk_icb_device {
   mtl_render_pipeline_state *encode_pipeline;
   simple_mtx_t lock;
   struct util_dynarray free_chunks; /* struct kk_icb_chunk */
};

/* Per command buffer: the chunks its draws use, until it's reset. Within a
 * render pass every call gets slots of its own: Metal doesn't say which stage
 * reads an executing ICB's commands, so no in-pass barrier can protect a slot
 * being rewritten. Passes end with a full barrier (end_encoder), so the next
 * pass starts over in the same chunks. */
struct kk_icb_cmd {
   struct util_dynarray chunks; /* struct kk_icb_chunk */
   uint32_t chunk;              /* the chunk the next call starts looking in */
   uint32_t next_slot;          /* its first free slot */
   uint32_t pass_slots;         /* slots this render pass has used */
};

/* Parameters of kk_icb_encode, uploaded per call (matches the MSL struct) */
struct kk_icb_encode_params {
   uint64_t icb;          /* address of icb_id below, as an ICBContainer */
   uint64_t args;         /* the app's draw arguments */
   uint64_t count;        /* the app's count */
   uint64_t cond;         /* conditional rendering value, 0 for none */
   uint64_t range;        /* MTLIndirectCommandBufferExecutionRange below */
   uint64_t index_buffer; /* 0 for non-indexed draws */
   uint32_t stride;
   uint32_t max_draws;
   uint32_t location;
   uint32_t mode;         /* 0 draws, 1 16-bit indexed, 2 32-bit indexed */
   uint32_t prim;         /* MTLPrimitiveType */
   uint32_t cond_inverted;
   uint32_t index_count;  /* indices in the bound range; draws are clamped to it */
   uint32_t pad;
   uint64_t icb_id;       /* the ICB's gpuResourceID */
   uint32_t range_location, range_length;
};

void kk_icb_device_init(struct kk_device *dev);
void kk_icb_device_finish(struct kk_device *dev);

/* `count` slots no other call in this render pass uses: the ICB and the
 * first slot. False when the pass is over its budget or out of memory. */
bool kk_icb_reserve(struct kk_cmd_buffer *cmd, uint32_t count,
                    struct kk_icb_chunk *chunk, uint32_t *location);

/* A new render pass may reuse every slot */
static inline void
kk_icb_cmd_new_pass(struct kk_icb_cmd *icb)
{
   icb->chunk = 0u;
   icb->next_slot = 0u;
   icb->pass_slots = 0u;
}

/* Back to the device's pool (the command buffer is no longer pending) */
void kk_icb_cmd_release(struct kk_device *dev, struct kk_icb_cmd *icb);

#endif
