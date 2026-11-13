/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_PHYSICAL_DEVICE_H
#define KK_PHYSICAL_DEVICE_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/simple_mtx.h"
#include "vk_physical_device.h"
#include "vk_sync.h"
#include "vk_sync_binary.h"

#include "wsi_common.h"

#include <sys/types.h>

struct kk_instance;
struct kk_physical_device;

struct kk_queue_family {
   VkQueueFlags queue_flags;
   uint32_t queue_count;
};

struct kk_memory_heap {
   uint64_t size;
   VkMemoryHeapFlags flags;
   uint64_t (*budget)(struct kk_physical_device *pdev);
   uint64_t (*used)(struct kk_physical_device *pdev);
};

struct kk_device_info {
   uint32_t max_workgroup_count[3];
   uint32_t max_workgroup_invocations;
   uint32_t max_compute_shared_memory_size;
   uint32_t gpu_apple_family;
   enum mtl_language_version msl_version;
   uint64_t max_buffer_size;
   uint32_t max_sampler_count;
   uint32_t rendering_tile_width;
   uint32_t rendering_tile_height;
   VkSampleCountFlags supported_sample_counts;
};

struct kk_env_settings {
   bool experimental_enabled;
   bool gpu_capture_enabled;
   /* KK_RAY_TRACING=1: expose ray tracing while it is in development */
   bool ray_tracing_enabled;
   /* KK_SPARSE=1: expose sparse memory while it is in development */
   bool sparse_enabled;
   /* KK_NVX=1: expose VK_NVX_binary_import and VK_NVX_image_view_handle for
    * the MetalFX upscaling work (logging stubs for now) */
   bool nvx_enabled;
   /* KK_QUEUE_COUNT=n (1 to 4): queues in the queue family. Streamline's
    * DLSS Frame Generation needs a second one; more than one changes how
    * games schedule work, so it's opt-in. */
   uint32_t queue_count;
   /* KK_COMPUTE_QUEUES=n[,m] (each 1 to 4): queue families after the main
    * one, compute and transfer only, with n (and m) queues. Engines look for
    * a separate family for async compute; DOOM: The Dark Ages also wants one
    * it doesn't use itself before it offers FSR frame generation (2,1). Off
    * by default, like KK_QUEUE_COUNT. */
   uint32_t compute_queue_counts[KK_MAX_COMPUTE_FAMILIES];
   /* KK_DEVICE_LOCAL_TYPE=0 drops the regular DEVICE_LOCAL-only memory type
    * (on by default; see kk_physical_device.c) */
   bool device_local_type;
   /* KK_PRIVATE_DEVICE_LOCAL=0 keeps that type in shared memory instead of
    * private (on by default; see kk_device_memory.c) */
   bool private_device_local;
   const char *gpu_capture_dir;
   uint64_t disabled_workarounds;
   /* KK_EXACT=1: no optimization that can change output (e.g. Metal's own
    * order for subgroup float sums); off by default */
   bool exact_output;
   /* KK_ATOMIC64_EMULATION=1: 64-bit atomics Metal lacks (compare-and-swap,
    * exchange, add, ...) emulated with a lock table; off by default */
   bool atomic64_emulation;
};

struct kk_physical_device {
   struct vk_physical_device vk;
   mtl_device *mtl_dev_handle;
   struct kk_device_info info;
   struct kk_env_settings settings;

   /* Software fp64 library, built on first use (kk_shader.c) */
   simple_mtx_t softfp64_mutex;
   struct nir_shader *softfp64;

   struct wsi_device wsi_device;

   uint8_t device_uuid[VK_UUID_SIZE];

   // TODO: add mapable VRAM heap if possible
   struct kk_memory_heap mem_heaps[3];
   VkMemoryType mem_types[4];
   uint8_t mem_heap_count;
   uint8_t mem_type_count;
   /* Private memory backing sparse resources, which only they may use; see
    * kk_sparse.c. mem_type_count when there is none. */
   uint8_t private_mem_type;
   /* The regular DEVICE_LOCAL-only type, UINT8_MAX when there is none */
   uint8_t device_local_mem_type;

   // Emulated binary sync type
   struct vk_sync_binary_type sync_binary_type;
   const struct vk_sync_type *sync_types[3];

   struct kk_queue_family queue_families[3];
   uint8_t queue_family_count;
};

static inline uint32_t
kk_min_cbuf_alignment()
{
   /* Size of vec4 */
   return 16;
}

VK_DEFINE_HANDLE_CASTS(kk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

static inline struct kk_instance *
kk_physical_device_instance(struct kk_physical_device *pdev)
{
   return (struct kk_instance *)pdev->vk.instance;
}

/* Memory types for everything but sparse resources and acceleration structure
 * storage (which can also use the private type) */
static inline uint32_t
kk_regular_mem_types(const struct kk_physical_device *pdev)
{
   return BITFIELD_MASK(pdev->mem_type_count) &
          ~BITFIELD_BIT(pdev->private_mem_type);
}

bool kk_extension_only_reported(const struct kk_physical_device *pdev,
                                const char *name);

VkResult kk_enumerate_physical_devices(struct vk_instance *_instance);
void kk_physical_device_destroy(struct vk_physical_device *vk_device);

#if defined(VK_USE_PLATFORM_WAYLAND_KHR) ||                                    \
   defined(VK_USE_PLATFORM_XCB_KHR) || defined(VK_USE_PLATFORM_XLIB_KHR) ||    \
   defined(VK_USE_PLATFORM_DISPLAY_KHR) || defined(VK_USE_PLATFORM_METAL_EXT)
#define KK_USE_WSI_PLATFORM
#endif

#endif // KK_PHYSICAL_DEVICE_H
