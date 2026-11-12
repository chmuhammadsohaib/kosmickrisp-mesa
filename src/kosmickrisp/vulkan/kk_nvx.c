/*
 * SPDX-License-Identifier: MIT
 */

/* VK_NVX_binary_import and VK_NVX_image_view_handle, behind KK_NVX=1.
 *
 * DLSS games require both extensions. Here they carry the private MetalFX
 * upscale channel of kk_mfx_channel.h: the only module accepted is the
 * upscaler, and its launches record a MetalFX upscale. CUDA binaries are
 * refused, since they can't run on Apple GPUs. A 64-bit image view handle is
 * the image view itself; the 32-bit one is its low half and only good for
 * logging.
 */

#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_image_view.h"
#include "kk_mfx.h"
#include "kk_mfx_channel.h"

#include "vk_log.h"
#include "vk_object.h"

#include "util/u_atomic.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

struct kk_cu_module {
   struct vk_object_base base;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_cu_module, base, VkCuModuleNVX,
                               VK_OBJECT_TYPE_CU_MODULE_NVX)

struct kk_cu_function {
   struct vk_object_base base;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_cu_function, base, VkCuFunctionNVX,
                               VK_OBJECT_TYPE_CU_FUNCTION_NVX)

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateCuModuleNVX(VkDevice device,
                     const VkCuModuleCreateInfoNVX *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkCuModuleNVX *pModule)
{
   VK_FROM_HANDLE(kk_device, dev, device);

   const size_t magic_len = strlen(KK_MFX_MODULE_MAGIC);
   if (!pCreateInfo->pData || pCreateInfo->dataSize < magic_len ||
       memcmp(pCreateInfo->pData, KK_MFX_MODULE_MAGIC, magic_len) != 0) {
      fprintf(stderr,
              "[kk] nvx: refusing a %zu-byte CUDA module (only the MetalFX "
              "upscaler runs here)\n",
              pCreateInfo->dataSize);
      return vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);
   }

   struct kk_cu_module *module = vk_object_zalloc(
      &dev->vk, pAllocator, sizeof(*module), VK_OBJECT_TYPE_CU_MODULE_NVX);
   if (!module)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pModule = kk_cu_module_to_handle(module);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyCuModuleNVX(VkDevice device, VkCuModuleNVX _module,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_cu_module, module, _module);

   if (module)
      vk_object_free(&dev->vk, pAllocator, module);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateCuFunctionNVX(VkDevice device,
                       const VkCuFunctionCreateInfoNVX *pCreateInfo,
                       const VkAllocationCallbacks *pAllocator,
                       VkCuFunctionNVX *pFunction)
{
   VK_FROM_HANDLE(kk_device, dev, device);

   if (!pCreateInfo->pName ||
       strcmp(pCreateInfo->pName, KK_MFX_FUNCTION_UPSCALE) != 0) {
      fprintf(stderr, "[kk] nvx: refusing unknown function \"%s\"\n",
              pCreateInfo->pName ? pCreateInfo->pName : "");
      return vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);
   }

   struct kk_cu_function *function =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*function),
                       VK_OBJECT_TYPE_CU_FUNCTION_NVX);
   if (!function)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pFunction = kk_cu_function_to_handle(function);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyCuFunctionNVX(VkDevice device, VkCuFunctionNVX _function,
                        const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_cu_function, function, _function);

   if (function)
      vk_object_free(&dev->vk, pAllocator, function);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdCuLaunchKernelNVX(VkCommandBuffer commandBuffer,
                        const VkCuLaunchInfoNVX *pLaunchInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   /* The only function is the upscaler; its parameter block is pParams */
   if ((pLaunchInfo->paramCount != sizeof(struct kk_mfx_upscale_params) &&
        pLaunchInfo->paramCount != KK_MFX_PARAMS_V1_SIZE) ||
       !pLaunchInfo->pParams) {
      static uint32_t warn_count;
      if (p_atomic_inc_return(&warn_count) <= 16)
         fprintf(stderr,
                 "[kk] nvx: ignoring a launch with %zu parameter bytes "
                 "(expected %zu)\n",
                 pLaunchInfo->paramCount,
                 sizeof(struct kk_mfx_upscale_params));
      return;
   }

   struct kk_mfx_upscale_params params = {0};
   memcpy(&params, pLaunchInfo->pParams, pLaunchInfo->paramCount);
   if (params.version == 1) {
      params.max_render_width = params.render_width;
      params.max_render_height = params.render_height;
      params.version = KK_MFX_PARAMS_VERSION;
   }
   kk_mfx_cmd_upscale(cmd, &params);
}

VKAPI_ATTR uint32_t VKAPI_CALL
kk_GetImageViewHandleNVX(VkDevice device,
                         const VkImageViewHandleInfoNVX *pInfo)
{
   return (uint32_t)(uintptr_t)kk_image_view_from_handle(pInfo->imageView);
}

VKAPI_ATTR uint64_t VKAPI_CALL
kk_GetImageViewHandle64NVX(VkDevice device,
                           const VkImageViewHandleInfoNVX *pInfo)
{
   return (uint64_t)(uintptr_t)kk_image_view_from_handle(pInfo->imageView);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_GetImageViewAddressNVX(VkDevice device, VkImageView imageView,
                          VkImageViewAddressPropertiesNVX *pProperties)
{
   /* Image views have no GPU address on Metal */
   pProperties->deviceAddress = 0;
   pProperties->size = 0;
   return VK_SUCCESS;
}
