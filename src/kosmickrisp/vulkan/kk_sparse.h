/*
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_SPARSE_H
#define KK_SPARSE_H 1

#include "kk_private.h"

struct kk_device;
struct kk_image;
struct kk_physical_device;
struct kk_queue;
struct vk_queue_submit;

bool kk_sparse_image_create_supported(
   const struct kk_physical_device *pdev,
   const VkPhysicalDeviceImageFormatInfo2 *info);

void kk_sparse_image_format_properties(
   struct kk_physical_device *pdev,
   const VkPhysicalDeviceSparseImageFormatInfo2 *info, uint32_t *count,
   VkSparseImageFormatProperties2 *properties);

VkResult kk_image_init_sparse(struct kk_device *dev, struct kk_image *image);
void kk_image_finish_sparse(struct kk_device *dev, struct kk_image *image);
uint64_t kk_image_sparse_size_B(const struct kk_image *image);
void kk_image_sparse_memory_requirements(const struct kk_image *image,
                                         VkSparseImageMemoryRequirements *req);

void kk_queue_bind_sparse(struct kk_queue *queue,
                          const struct vk_queue_submit *submit);

#endif /* KK_SPARSE_H */
