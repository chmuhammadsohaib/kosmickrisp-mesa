/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_sync.h"

#include "kk_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/u_atomic.h"

#include <inttypes.h>
#include <stdio.h>

static VkResult
kk_timeline_init(struct vk_device *device, struct vk_sync *sync,
                 uint64_t initial_value)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);

   struct kk_device *dev = container_of(device, struct kk_device, vk);
   timeline->mtl_handle = mtl_new_shared_event(dev->mtl_handle);
   mtl_shared_event_set_signaled_value(timeline->mtl_handle, initial_value);

   return VK_SUCCESS;
}

static void
kk_timeline_finish(struct vk_device *device, struct vk_sync *sync)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   mtl_release(timeline->mtl_handle);
}

static VkResult
kk_timeline_signal(struct vk_device *device, struct vk_sync *sync,
                   uint64_t value)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   mtl_shared_event_set_signaled_value(timeline->mtl_handle, value);
   return VK_SUCCESS;
}

static VkResult
kk_timeline_get_value(struct vk_device *device, struct vk_sync *sync,
                      uint64_t *value)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   *value = mtl_shared_event_get_signaled_value(timeline->mtl_handle);
   return VK_SUCCESS;
}

static VkResult
kk_timeline_wait(struct vk_device *device, struct vk_sync *sync,
                 uint64_t wait_value, enum vk_sync_wait_flags wait_flags,
                 uint64_t abs_timeout_ns)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);

   /* abs_timeout_ns is the point in time when we should stop waiting, not the
    * absolute time in ns. Therefore, we need to compute the delta from now to
    * when we should stop waiting and convert to ms for Metal to be happy
    * (Similar to what dzn does).
    */
   uint64_t timeout_ms = 0u;
   if (abs_timeout_ns == OS_TIMEOUT_INFINITE) {
      timeout_ms = OS_TIMEOUT_INFINITE;
   } else {
      uint64_t cur_time = os_time_get_nano();
      uint64_t rel_timeout_ns =
         abs_timeout_ns > cur_time ? abs_timeout_ns - cur_time : 0;

      timeout_ms =
         (rel_timeout_ns / 1000000) + (rel_timeout_ns % 1000000 ? 1 : 0);
   }
   struct kk_device *dev = container_of(device, struct kk_device, vk);
   if (unlikely(dev->sync_debug.enabled) &&
       abs_timeout_ns == OS_TIMEOUT_INFINITE) {
      /* Wait in slices and say what the queue has done whenever one runs out:
       * whether the GPU still has work (or failed), or nothing that would
       * signal this value was ever submitted. */
      const uint64_t slice_ms = 5000u;
      for (uint32_t stalled = 1u;; stalled++) {
         if (mtl_shared_event_wait_until_signaled_value(timeline->mtl_handle,
                                                        wait_value, slice_ms))
            return VK_SUCCESS;
         const uint64_t committed = p_atomic_read(&dev->sync_debug.committed);
         const uint64_t completed = p_atomic_read(&dev->sync_debug.completed);
         fprintf(stderr,
                 "[kk] sync-debug: wait stalled %us on timeline %p for %" PRIu64
                 ", signaled %" PRIu64 "; command buffers committed %" PRIu64
                 ", completed %" PRIu64 " (%" PRIu64
                 " in flight), last GPU error %d\n",
                 stalled * (unsigned)(slice_ms / 1000u), (void *)timeline,
                 wait_value,
                 mtl_shared_event_get_signaled_value(timeline->mtl_handle),
                 committed, completed, committed - completed,
                 dev->sync_debug.last_error);
      }
   }

   int completed = mtl_shared_event_wait_until_signaled_value(
      timeline->mtl_handle, wait_value, timeout_ms);

   return completed != 0 ? VK_SUCCESS : VK_TIMEOUT;
}

const struct vk_sync_type kk_sync_type = {
   .size = sizeof(struct kk_sync_timeline),
   .features = VK_SYNC_FEATURE_TIMELINE | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_PENDING |
               VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL,
   .init = kk_timeline_init,
   .finish = kk_timeline_finish,
   .signal = kk_timeline_signal,
   .get_value = kk_timeline_get_value,
   .reset = NULL,
   .move = NULL,
   .wait = kk_timeline_wait,
   .wait_many = NULL,
   .import_opaque_fd = NULL,
   .export_opaque_fd = NULL,
   .import_sync_file = NULL,
   .export_sync_file = NULL,
   .import_win32_handle = NULL,
   .export_win32_handle = NULL,
   .set_win32_export_params = NULL,
};
