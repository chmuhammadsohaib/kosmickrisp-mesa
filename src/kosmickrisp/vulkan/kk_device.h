/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_DEVICE_H
#define KK_DEVICE_H 1

#include "kk_private.h"

#include "kk_mfx.h"
#include "kk_query_table.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "kosmickrisp/clc/kk_precompiled_shader.h"
#include "libkk_shaders.h"

#include "util/u_dynarray.h"

#include "vk_device.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct kk_bo;
struct kk_physical_device;
struct vk_pipeline_cache;

struct kk_residency_set {
   simple_mtx_t mutex;
   mtl_residency_set *handle;
   bool dirty; /* allocations added or removed since the last commit */
};

struct mtl_sampler_packed {
   enum mtl_sampler_address_mode mode_u;
   enum mtl_sampler_address_mode mode_v;
   enum mtl_sampler_address_mode mode_w;
   enum mtl_sampler_border_color border_color;

   enum mtl_sampler_min_mag_filter min_filter;
   enum mtl_sampler_min_mag_filter mag_filter;
   enum mtl_sampler_mip_filter mip_filter;
   enum mtl_sampler_reduction_mode reduction_mode;

   enum mtl_compare_function compare_func;
   float min_lod;
   float max_lod;
   uint32_t max_anisotropy;
   bool normalized_coordinates;
};

struct kk_rc_sampler {
   struct mtl_sampler_packed key;

   mtl_sampler *handle;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;
};

struct kk_sampler_heap {
   simple_mtx_t lock;

   struct kk_query_table table;

   /* Map of mtl_sampler_packed to kk_rc_sampler */
   struct hash_table *ht;
};

struct kk_precompiled_cache {
   struct kk_precompiled_shader shaders[LIBKK_NUM_PROGRAMS];
};

struct kk_device {
   struct vk_device vk;

   mtl_device *mtl_handle;
   mtl_compiler *mtl_compiler_handle;

   /* Dispatch table exposed to the user. Required since we need to record all
    * commands due to Metal limitations */
   struct vk_device_dispatch_table exposed_dispatch_table;

   struct kk_sampler_heap samplers;
   struct kk_query_table occlusion_queries;

   /* Track all heaps the user allocated so we can set them all as resident when
    * recording as required by Metal. */
   struct kk_residency_set residency_set;
   /* Depth/stencil states for dynamic depth/stencil, by state; kept for the
    * device's life (a game uses a few dozen at most) */
   struct {
      simple_mtx_t mutex;
      struct hash_table *table;
   } ds_states;

   struct kk_precompiled_cache precompiled_cache;

   bool has_queue;
   struct kk_queue queue;
   /* Queues 1 and up of the family (KK_QUEUE_COUNT); internal work and
    * presentation use queue */
   uint32_t extra_queue_count;
   struct kk_queue extra_queues[KK_MAX_EXTRA_QUEUES];

   struct vk_meta_device meta;

   /* 8-byte slots holding acceleration structure resource IDs. A slot's GPU
    * address is the acceleration structure's device address, so slots never
    * move. See kk_acceleration_structure.c */
   struct {
      simple_mtx_t mutex;
      struct util_dynarray bos;  /* struct kk_bo *, KK_AS_SLOTS_PER_BO each */
      struct util_dynarray free; /* uint32_t slot indices */
      uint32_t count;
   } as_slots;

   /* Private heaps for acceleration structures that can't be placed in their
    * buffer's memory, and the update policy (KK_RT_REFIT_LIMIT,
    * KK_RT_REFIT_MIN_PRIMS). See kk_acceleration_structure.c */
   struct {
      simple_mtx_t mutex;
      struct util_dynarray chunks; /* struct kk_as_chunk * */
      int64_t refit_limit;
      int64_t refit_min_prims;
   } as_pool;

   /* Ray tracing pipelines: the dispatch kernel and default hit group
    * functions, built on first use, and each pipeline's Metal linking. See
    * kk_rt_pipeline.c */
   struct {
      simple_mtx_t mutex;
      mtl_library *library;
      mtl_function_descriptor *kernel;
      mtl_binary_function *default_triangle;
      mtl_binary_function *default_box;
      uint64_t default_triangle_id;
      uint64_t default_box_id;
      struct hash_table *links; /* struct vk_pipeline * -> kk_rt_link */
      /* Visible function table slots of ray tracing shaders */
      struct util_dynarray free_slots;
      uint32_t next_slot;
   } rt;

   /* Ray tracing stacks not used by a command buffer; see kk_rt_stack_acquire */
   struct {
      simple_mtx_t mutex;
      struct util_dynarray free; /* struct kk_bo * */
   } rt_stacks;

   /* MetalFX upscaling for the NVX channel; see kk_mfx.c */
   struct kk_mfx_state mfx;

   /* KK_RT_STATS, NULL when off; see kk_rt_stats.c */
   struct kk_rt_stats *rt_stats;

   /* KK_SYNC_DEBUG: CPU waits that stall log what the queue has done; see
    * kk_timeline_wait */
   struct {
      bool enabled;
      uint64_t committed;
      uint64_t completed;
      int last_error;
   } sync_debug;

   /* Sparse memory, for unmapping pages; see kk_sparse.c */
   struct kk_bo *sparse_unmap_bo;
   /* KK_ATOMIC64_EMULATION: 32-bit locks, zeroed (kk_nir_lower_atomic64) */
   struct kk_bo *atomic64_locks;

   /* Geomtry heap */
   struct kk_bo *heap;
   util_once_flag heap_init_once;
};

VK_DEFINE_HANDLE_CASTS(kk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct kk_physical_device *
kk_device_physical(const struct kk_device *dev)
{
   return (struct kk_physical_device *)dev->vk.physical;
}

VkResult kk_device_init_meta(struct kk_device *dev);
void kk_device_finish_meta(struct kk_device *dev);
VkResult kk_device_init_lib(struct kk_device *dev);
void kk_device_finish_lib(struct kk_device *dev);
void kk_device_add_heap_to_residency_set(struct kk_device *dev, mtl_heap *heap);
void kk_device_remove_heap_from_residency_set(struct kk_device *dev,
                                              mtl_heap *heap);
void kk_device_add_buffer_to_residency_set(struct kk_device *dev,
                                           mtl_buffer *buffer);
void kk_device_remove_buffer_from_residency_set(struct kk_device *dev,
                                                mtl_buffer *buffer);
void kk_device_make_resources_resident(struct kk_device *dev);
/* Key of the depth/stencil state cache (kk_device_ds_state in kk_shader.c) */
struct kk_ds_key {
   VkCompareOp depth_compare_op;
   uint8_t stencil[2][6];
   bool has_depth_write;
   bool has_stencil_test;
};

struct kk_shader_info;
mtl_depth_stencil_state *kk_device_ds_state(struct kk_device *dev,
                                            const struct kk_shader_info *info);

void kk_device_init_rt(struct kk_device *dev);
void kk_device_finish_rt(struct kk_device *dev);
void kk_device_finish_rt_stacks(struct kk_device *dev);
/* Ray tracing stacks, reused across command buffers */
struct kk_bo *kk_rt_stack_acquire(struct kk_device *dev, uint64_t size_B);
void kk_rt_stack_release(struct kk_device *dev, struct kk_bo *bo);

void kk_device_init_as_slots(struct kk_device *dev);
void kk_device_finish_as_slots(struct kk_device *dev);

/* Required to create a sampler */
mtl_sampler *kk_sampler_create(struct kk_device *dev,
                               const struct mtl_sampler_packed *packed);
VkResult kk_sampler_heap_add(struct kk_device *dev,
                             struct mtl_sampler_packed desc,
                             struct kk_rc_sampler **out);
void kk_sampler_heap_remove(struct kk_device *dev, struct kk_rc_sampler *rc);

#endif // KK_DEVICE_H
