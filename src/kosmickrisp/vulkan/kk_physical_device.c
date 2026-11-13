/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_physical_device.h"

#include "kk_log_queries.h"

#include "kk_acceleration_structure.h"

#include "kk_debug.h"
#include "kk_entrypoints.h"
#include "kk_image.h"
#include "kk_instance.h"
#include "kk_nir_lower_vbo.h"
#include "kk_sync.h"
#include "kk_wsi.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/ns_process_info.h"
#include "kosmickrisp/compiler/nir_to_msl.h"

#include "util/disk_cache.h"
#include "util/mesa-blake3.h"
#include "git_sha1.h"

#include "vulkan/wsi/wsi_common.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_drm_syncobj.h"
#include "vk_shader_module.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

/* kosmickrisp-compat-switches: KK_VENDOR_ID sets the GPU identity the game sees. apple (default)
 * is the real one; nvidia and amd are presets with a matching name and a driver version in that
 * vendor's own format, so driver-version checks parse sensibly; or raw hex
 * "vendor[:device[:driverVersion]]". Capabilities don't change. */
static void
kk_apply_vendor_identity(struct vk_properties *properties)
{
   const char *value = getenv("KK_VENDOR_ID");
   if (!value || !value[0] || strcmp(value, "apple") == 0)
      return;
   if (strcmp(value, "nvidia") == 0) {
      properties->vendorID = 0x10de;
      properties->deviceID = 0x2786;          /* GeForce RTX 4070 */
      properties->driverVersion = 0x8C178000; /* 560.94 in NVIDIA's version packing */
      snprintf(properties->deviceName, sizeof(properties->deviceName), "NVIDIA GeForce RTX 4070");
      return;
   }
   if (strcmp(value, "amd") == 0) {
      properties->vendorID = 0x1002;
      properties->deviceID = 0x73bf;          /* Radeon RX 6800 XT */
      properties->driverVersion = VK_MAKE_API_VERSION(0, 2, 0, 302);
      snprintf(properties->deviceName, sizeof(properties->deviceName), "AMD Radeon RX 6800 XT");
      return;
   }
   unsigned vendor = 0, device = 0, version = 0;
   int count = sscanf(value, "%x:%x:%x", &vendor, &device, &version);
   if (count >= 1 && vendor)
      properties->vendorID = vendor;
   if (count >= 2)
      properties->deviceID = device;
   if (count >= 3)
      properties->driverVersion = version;
}

/* kosmickrisp-compat-switches: KK_DRIVER_ID sets the driver identity the game sees
 * (VkPhysicalDeviceDriverProperties and driverVersion). radv, amd and nvidia are presets with
 * that driver's name, info and a driverVersion in its own format; a number sets only driverID.
 * Applied after KK_VENDOR_ID, so a preset's driverVersion wins. Capabilities don't change. */
static void
kk_apply_driver_identity(struct vk_properties *properties)
{
   const char *value = getenv("KK_DRIVER_ID");
   if (!value || !value[0])
      return;
   if (strcmp(value, "radv") == 0) {
      properties->driverID = VK_DRIVER_ID_MESA_RADV;
      properties->driverVersion = VK_MAKE_API_VERSION(0, 24, 2, 0); /* Mesa 24.2.0 */
      snprintf(properties->driverName, VK_MAX_DRIVER_NAME_SIZE, "radv");
      snprintf(properties->driverInfo, VK_MAX_DRIVER_INFO_SIZE, "Mesa 24.2.0");
      return;
   }
   if (strcmp(value, "amd") == 0) {
      properties->driverID = VK_DRIVER_ID_AMD_PROPRIETARY;
      properties->driverVersion = VK_MAKE_API_VERSION(0, 2, 0, 302);
      snprintf(properties->driverName, VK_MAX_DRIVER_NAME_SIZE, "AMD proprietary driver");
      snprintf(properties->driverInfo, VK_MAX_DRIVER_INFO_SIZE, "24.10.1");
      return;
   }
   if (strcmp(value, "nvidia") == 0) {
      properties->driverID = VK_DRIVER_ID_NVIDIA_PROPRIETARY;
      properties->driverVersion = 0x8C178000; /* 560.94 in NVIDIA's version packing */
      snprintf(properties->driverName, VK_MAX_DRIVER_NAME_SIZE, "NVIDIA");
      snprintf(properties->driverInfo, VK_MAX_DRIVER_INFO_SIZE, "560.94");
      return;
   }
   char *end;
   unsigned long id = strtoul(value, &end, 0);
   if (end != value && *end == '\0')
      properties->driverID = (VkDriverId)id;
}

/* kosmickrisp-compat-switches: KK_REPORT_EXTENSIONS=<comma list> makes
 * vkEnumerateDeviceExtensionProperties also list those device extensions. They are advertised
 * only: vkCreateDevice leaves them out with a warning (kk_extension_only_reported). */
static bool
kk_reported_extension_listed(const char *name)
{
   const char *value = getenv("KK_REPORT_EXTENSIONS");
   if (!value || !value[0])
      return false;

   const size_t len = strlen(name);
   for (const char *c = value; *c;) {
      c += strspn(c, ", ");
      const size_t n = strcspn(c, ", ");
      if (n == len && strncmp(c, name, n) == 0)
         return true;
      c += n;
   }
   return false;
}

static int
kk_device_extension_index(const char *name)
{
   for (int i = 0; i < VK_DEVICE_EXTENSION_COUNT; i++) {
      if (strcmp(vk_device_extensions[i].extensionName, name) == 0)
         return i;
   }
   return -1;
}

bool
kk_extension_only_reported(const struct kk_physical_device *pdev,
                           const char *name)
{
   const int index = kk_device_extension_index(name);
   const bool supported =
      index >= 0 && pdev->vk.supported_extensions.extensions[index];
   return !supported && kk_reported_extension_listed(name);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                      const char *pLayerName,
                                      uint32_t *pPropertyCount,
                                      VkExtensionProperties *pProperties)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);
   const char *value = getenv("KK_REPORT_EXTENSIONS");

   if (pLayerName || !value || !value[0])
      return vk_common_EnumerateDeviceExtensionProperties(
         physicalDevice, pLayerName, pPropertyCount, pProperties);

   VK_OUTARRAY_MAKE_TYPED(VkExtensionProperties, out, pProperties,
                          pPropertyCount);

   for (int i = 0; i < VK_DEVICE_EXTENSION_COUNT; i++) {
      if (!pdev->vk.supported_extensions.extensions[i])
         continue;
      vk_outarray_append_typed(VkExtensionProperties, &out, prop)
         *prop = vk_device_extensions[i];
   }

   char list[1024];
   snprintf(list, sizeof(list), "%s", value);
   for (char *saved = NULL, *name = strtok_r(list, ", ", &saved); name;
        name = strtok_r(NULL, ", ", &saved)) {
      if (!kk_extension_only_reported(pdev, name))
         continue;
      const int index = kk_device_extension_index(name);
      vk_outarray_append_typed(VkExtensionProperties, &out, prop)
      {
         *prop = (VkExtensionProperties){
            .specVersion = index >= 0 ? vk_device_extensions[index].specVersion : 1,
         };
         snprintf(prop->extensionName, sizeof(prop->extensionName), "%s", name);
      }
   }

   return vk_outarray_status(&out);
}

static uint32_t
kk_get_vk_version()
{
   /* Version override takes priority */
   const uint32_t version_override = vk_get_version_override();
   if (version_override)
      return version_override;

   return VK_MAKE_VERSION(1, 4, VK_HEADER_VERSION);
}

static void
kk_get_device_extensions(const struct kk_instance *instance,
                         const struct kk_physical_device *pdev,
                         struct vk_device_extension_table *ext)
{
   *ext = (struct vk_device_extension_table){
      /* Vulkan 1.1 */
      .KHR_16bit_storage = true,
      .KHR_bind_memory2 = true,
      .KHR_dedicated_allocation = true,
      .KHR_descriptor_update_template = true,
      .KHR_device_group = true,
      .KHR_external_fence = true,
      .KHR_external_memory = true,
      .KHR_external_semaphore = true,
      .KHR_get_memory_requirements2 = true,
      .KHR_maintenance1 = true,
      .KHR_maintenance2 = true,
      .KHR_maintenance3 = true,
      .KHR_multiview = true,
      .KHR_relaxed_block_layout = true,
      .KHR_sampler_ycbcr_conversion = true,
      .KHR_shader_draw_parameters = true,
      .KHR_storage_buffer_storage_class = true,
      .KHR_variable_pointers = true,

      /* Vulkan 1.2 */
      .KHR_8bit_storage = true,
      .KHR_buffer_device_address = true, /* Required in Vulkan 1.3 */
      .KHR_create_renderpass2 = true,
      .KHR_depth_stencil_resolve = true,
      .KHR_draw_indirect_count = true,
      .KHR_driver_properties = true,
      .KHR_image_format_list = true,
      .KHR_imageless_framebuffer = true,
      .KHR_sampler_mirror_clamp_to_edge = true,
      .KHR_separate_depth_stencil_layouts = true,
      .KHR_shader_atomic_int64 = false,
      .KHR_shader_float_controls = true,
      .KHR_shader_float16_int8 = true,
      .KHR_shader_subgroup_extended_types = true,
      .KHR_spirv_1_4 = true,
      .KHR_timeline_semaphore = true,
      .KHR_uniform_buffer_standard_layout = true,
      .KHR_vulkan_memory_model = true, /* Required in Vulkan 1.3 */
      .EXT_buffer_device_address = true,
      .EXT_descriptor_indexing = true,
      .EXT_host_query_reset = true,
      .EXT_sampler_filter_minmax = pdev->info.gpu_apple_family >= 10,
      .EXT_scalar_block_layout = true,
      .EXT_separate_stencil_usage = true,
      .EXT_shader_viewport_index_layer = true,

      /* Vulkan 1.3 */
      .KHR_copy_commands2 = true,
      .KHR_dynamic_rendering = true,
      .KHR_format_feature_flags2 = true,
      .KHR_maintenance4 = true,
      .KHR_shader_integer_dot_product = true,
      .KHR_shader_non_semantic_info = true,
      .KHR_shader_terminate_invocation = true,
      .KHR_synchronization2 = true,
      .KHR_zero_initialize_workgroup_memory = true,
      .EXT_4444_formats = true,
      .EXT_extended_dynamic_state = true,
      .EXT_extended_dynamic_state2 = true,
      .EXT_image_robustness = true,
      .EXT_inline_uniform_block = true,
      .EXT_pipeline_creation_cache_control = true,
      .EXT_pipeline_creation_feedback = true,
      .EXT_private_data = true,
      .EXT_shader_demote_to_helper_invocation = true,
      .EXT_shader_stencil_export = true,
      .EXT_subgroup_size_control = true,
      .EXT_texel_buffer_alignment = true,
      .EXT_texture_compression_astc_hdr = true,
      .EXT_tooling_info = true,
      .EXT_ycbcr_2plane_444_formats = true,

      /* Vulkan 1.4 */
      .KHR_dynamic_rendering_local_read = true,
      .KHR_global_priority = true,
      .KHR_line_rasterization = true,
      .KHR_index_type_uint8 = true,
      .KHR_load_store_op_none = true,
      .KHR_maintenance5 = true,
      .KHR_maintenance6 = true,
      .KHR_map_memory2 = true,
      .KHR_push_descriptor = true,
      .KHR_shader_expect_assume = true,
      .KHR_shader_float_controls2 = true,
      .KHR_shader_subgroup_rotate = true,
      .KHR_vertex_attribute_divisor = true,
      .EXT_global_priority = true,
      .EXT_global_priority_query = true,
      .EXT_host_image_copy = true,
      .EXT_index_type_uint8 = true,
      .EXT_line_rasterization = true,
      .EXT_pipeline_robustness = true,
      .EXT_vertex_attribute_divisor = true,

      /* Optional extensions */
      .KHR_calibrated_timestamps = true,
      .KHR_deferred_host_operations = true,
      .KHR_maintenance7 = true,
      .KHR_maintenance8 = true,
      .KHR_maintenance9 = true,
      .KHR_maintenance10 = true,
#ifdef KK_USE_WSI_PLATFORM
      .KHR_present_id = true,
      .KHR_present_id2 = true,
      .KHR_present_wait = true,
      .KHR_present_wait2 = true,
#endif
      .KHR_robustness2 = true,
      .KHR_shader_fma = true,
      .KHR_shader_maximal_reconvergence = true,
      .KHR_shader_relaxed_extended_instruction = true,
      .KHR_shader_subgroup_uniform_control_flow = true,
      .KHR_shader_untyped_pointers = true,
#ifdef KK_USE_WSI_PLATFORM
      .KHR_swapchain = true,
      .KHR_swapchain_maintenance1 = true,
      .KHR_swapchain_mutable_format = true,
#endif
      .KHR_unified_image_layouts = true,
      .KHR_workgroup_memory_explicit_layout = true,

      .EXT_attachment_feedback_loop_layout = true,
      .EXT_attachment_feedback_loop_dynamic_state = true,
      .EXT_blend_operation_advanced = true,
      .EXT_border_color_swizzle = KK_EXPERIMENTAL(CUSTOM_BORDER),
      .EXT_calibrated_timestamps = true,
      .EXT_conditional_rendering = true,
      .EXT_custom_border_color = KK_EXPERIMENTAL(CUSTOM_BORDER),
      .EXT_custom_resolve = true,
      .EXT_debug_marker = true,
      .EXT_depth_clip_control = true,
      .EXT_depth_clip_enable = true,
      .EXT_extended_dynamic_state3 = true,
      .EXT_external_memory_metal = true,
      .EXT_external_memory_host = true,
      .EXT_hdr_metadata = true,
      .EXT_image_2d_view_of_3d = true,
      .EXT_image_view_min_lod = KK_EXPERIMENTAL(IMAGE_VIEW_MIN_LOD),
      .EXT_load_store_op_none = true,
      .EXT_map_memory_placed = true,
      .EXT_memory_budget = true,
      .EXT_multi_draw = true,
      .EXT_mutable_descriptor_type = true,
      .EXT_nested_command_buffer = true,
      .EXT_post_depth_coverage = true,
      .EXT_primitive_restart_index = true,
      .EXT_primitive_topology_list_restart = true,
      .EXT_provoking_vertex = true,
      .EXT_robustness2 = true,
      .EXT_sample_locations = true,
      .EXT_shader_atomic_float = true,
      .EXT_shader_replicated_composites = true,
      .EXT_shader_subgroup_ballot = true,
      .EXT_shader_subgroup_vote = true,
#ifdef KK_USE_WSI_PLATFORM
      .EXT_swapchain_maintenance1 = true,
#endif
      .EXT_vertex_attribute_robustness = true,

      .GOOGLE_decorate_string = true,
      .GOOGLE_hlsl_functionality1 = true,
      .GOOGLE_user_type = true,
      .KHR_external_fence_fd = true,
      .KHR_external_semaphore_fd = true,

      .AMD_shader_image_load_store_lod = true,
      .AMD_buffer_marker = true,

      /* kosmickrisp-ray-tracing: in development, off unless KK_RAY_TRACING=1 */
      .KHR_acceleration_structure = pdev->settings.ray_tracing_enabled,
      .KHR_ray_query = pdev->settings.ray_tracing_enabled,
      .KHR_ray_tracing_pipeline = pdev->settings.ray_tracing_enabled,
      .KHR_pipeline_library = pdev->settings.ray_tracing_enabled,
      .EXT_pipeline_library_group_handles = pdev->settings.ray_tracing_enabled,

      /* kosmickrisp-metalfx: in development, off unless KK_NVX=1 */
      .NVX_binary_import = pdev->settings.nvx_enabled,
      .NVX_image_view_handle = pdev->settings.nvx_enabled,
   };
}

static void
kk_get_device_features(
   const struct kk_physical_device *pdev,
   const struct vk_device_extension_table *supported_extensions,
   struct vk_features *features)
{
   *features = (struct vk_features){
      /* Vulkan 1.0 */
      .alphaToOne = true,
      .depthBiasClamp = true,
      .depthBounds = pdev->info.gpu_apple_family >= 10,
      .depthClamp = true,
      .drawIndirectFirstInstance = true,
      .dualSrcBlend = true,
      .fragmentStoresAndAtomics = true,
      .fullDrawIndexUint32 = true,
      .imageCubeArray = true,
      .independentBlend = true,
      .inheritedQueries = true,
      .largePoints = true,
      .logicOp = true,
      .multiViewport = true,
      .occlusionQueryPrecise = true,
      .robustBufferAccess = true,
      .samplerAnisotropy = true,
      .sampleRateShading = true,
      .shaderClipDistance = true,
      .shaderCullDistance = true,
      .shaderImageGatherExtended = true,
      .shaderInt16 = true,
      .shaderInt64 = true,
      .shaderResourceMinLod = true,
      .shaderSampledImageArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderStorageImageArrayDynamicIndexing = true,
      .shaderStorageImageExtendedFormats = true,
      .shaderStorageImageReadWithoutFormat = true,
      .shaderStorageImageWriteWithoutFormat = true,
      .shaderTessellationAndGeometryPointSize = true,
      .shaderUniformBufferArrayDynamicIndexing = true,
      .tessellationShader = true,
      .textureCompressionASTC_LDR = true,
      .textureCompressionBC = true,
      .textureCompressionETC2 = true,
      .vertexPipelineStoresAndAtomics = true,

      /* Vulkan 1.1 */
      .multiview = true,
      .samplerYcbcrConversion = true,
      .shaderDrawParameters = true,
      .storageBuffer16BitAccess = true,
      .storageInputOutput16 = true,
      .storagePushConstant16 = true,
      .uniformAndStorageBuffer16BitAccess = true,
      .variablePointers = true,
      .variablePointersStorageBuffer = true,

      /* Vulkan 1.2 */
      .descriptorBindingInlineUniformBlockUpdateAfterBind = true,
      .descriptorBindingPartiallyBound = true,
      .descriptorBindingSampledImageUpdateAfterBind = true,
      .descriptorBindingStorageBufferUpdateAfterBind = true,
      .descriptorBindingStorageImageUpdateAfterBind = true,
      .descriptorBindingStorageTexelBufferUpdateAfterBind = true,
      .descriptorBindingUniformBufferUpdateAfterBind = true,
      .descriptorBindingUniformTexelBufferUpdateAfterBind = true,
      .descriptorBindingUpdateUnusedWhilePending = true,
      .descriptorBindingVariableDescriptorCount = true,
      .descriptorIndexing = true,
      .drawIndirectCount = true,
      .hostQueryReset = true,
      .imagelessFramebuffer = true,
      .multiDrawIndirect = true,
      .runtimeDescriptorArray = true,
      .samplerFilterMinmax = supported_extensions->EXT_sampler_filter_minmax,
      .samplerMirrorClampToEdge = true,
      .scalarBlockLayout = true,
      .separateDepthStencilLayouts = true,
      .shaderFloat16 = true,
      .shaderInputAttachmentArrayDynamicIndexing = true,
      .shaderInputAttachmentArrayNonUniformIndexing = true,
      .shaderInt8 = true,
      .shaderOutputLayer = true,
      .shaderOutputViewportIndex = true,
      .shaderSampledImageArrayNonUniformIndexing = true,
      .shaderStorageBufferArrayNonUniformIndexing = true,
      .shaderStorageImageArrayNonUniformIndexing = true,
      .shaderStorageTexelBufferArrayDynamicIndexing = true,
      .shaderStorageTexelBufferArrayNonUniformIndexing = true,
      .shaderSubgroupExtendedTypes = true,
      .shaderUniformBufferArrayNonUniformIndexing = true,
      .shaderUniformTexelBufferArrayDynamicIndexing = true,
      .shaderUniformTexelBufferArrayNonUniformIndexing = true,
      .storageBuffer8BitAccess = true,
      .storagePushConstant8 = true,
      .subgroupBroadcastDynamicId = true,
      .timelineSemaphore = true,
      .uniformAndStorageBuffer8BitAccess = true,
      .uniformBufferStandardLayout = true,

      /* Sparse memory: buffers and 2D images (kk_sparse.c) */
      .sparseBinding = pdev->settings.sparse_enabled,
      .sparseResidencyBuffer = pdev->settings.sparse_enabled,
      .sparseResidencyImage2D = pdev->settings.sparse_enabled,

      /* VK_KHR_acceleration_structure */
      .accelerationStructure = supported_extensions->KHR_acceleration_structure,
      .accelerationStructureCaptureReplay = false,
      .accelerationStructureIndirectBuild = false,
      .accelerationStructureHostCommands = false,
      .descriptorBindingAccelerationStructureUpdateAfterBind =
         supported_extensions->KHR_acceleration_structure,

      /* VK_KHR_ray_query */
      .rayQuery = supported_extensions->KHR_ray_query,

      /* VK_EXT_pipeline_library_group_handles: group handles only depend on
       * their shaders (kk_rt_pipeline.c) */
      .pipelineLibraryGroupHandles =
         supported_extensions->EXT_pipeline_library_group_handles,

      /* VK_KHR_ray_tracing_pipeline */
      .rayTracingPipeline = supported_extensions->KHR_ray_tracing_pipeline,
      .rayTracingPipelineShaderGroupHandleCaptureReplay = false,
      .rayTracingPipelineShaderGroupHandleCaptureReplayMixed = false,
      .rayTracingPipelineTraceRaysIndirect =
         supported_extensions->KHR_ray_tracing_pipeline,
      .rayTraversalPrimitiveCulling =
         supported_extensions->KHR_ray_tracing_pipeline,

      /* Vulkan 1.3 */
      .bufferDeviceAddress = true,
      .computeFullSubgroups = true,
      .dynamicRendering = true,
      .extendedDynamicState = true,
      .extendedDynamicState2 = true,
      .extendedDynamicState2LogicOp = false,
      .extendedDynamicState2PatchControlPoints = true,
      .inlineUniformBlock = true,
      .maintenance4 = true,
      .pipelineCreationCacheControl = true,
      .privateData = true,
      .robustImageAccess = true,
      .shaderDemoteToHelperInvocation = true,
      .shaderIntegerDotProduct = true,
      .shaderTerminateInvocation = true,
      .shaderZeroInitializeWorkgroupMemory = true,
      .subgroupSizeControl = true,
      .synchronization2 = true,
      .texelBufferAlignment = true,
      .textureCompressionASTC_HDR = true,
      .vulkanMemoryModel = true,
      .vulkanMemoryModelDeviceScope = true,

      /* Vulkan 1.4 */
      .bresenhamLines = true,
      .dynamicRenderingLocalRead = true,
      .globalPriorityQuery = true,
      .hostImageCopy = true,
      .indexTypeUint8 = true,
      .maintenance5 = true,
      .maintenance6 = true,
      .pipelineRobustness = true,
      .pushDescriptor = true,
      .shaderFloatControls2 = true,
      .shaderSubgroupRotate = true,
      .shaderSubgroupRotateClustered = true,
      .vertexAttributeInstanceRateDivisor = true,
      .vertexAttributeInstanceRateZeroDivisor = true,

      /* VK_EXT_mutable_descriptor_type */
      .mutableDescriptorType = true,

      /* VK_KHR_maintenance7 */
      .maintenance7 = true,

      /* VK_KHR_maintenance8 */
      .maintenance8 = true,

      /* VK_KHR_maintenance9 */
      .maintenance9 = true,

      /* VK_KHR_maintenance10 */
      .maintenance10 = true,

#ifdef KK_USE_WSI_PLATFORM
      /* VK_KHR_present_id */
      .presentId = true,

      /* VK_KHR_present_id2 */
      .presentId2 = true,

      /* VK_KHR_present_wait */
      .presentWait = true,

      /* VK_KHR_present_wait2 */
      .presentWait2 = true,
#endif

      /* VK_KHR_robustness2 */
      .robustBufferAccess2 = true,
      .robustImageAccess2 = true,
      .nullDescriptor = true,

      /* VK_KHR_shader_expect_assume */
      .shaderExpectAssume = true,

      /* VK_KHR_shader_fma */
      .shaderFmaFloat16 = true,
      .shaderFmaFloat32 = true,
      .shaderFmaFloat64 = false,

      /* VK_KHR_shader_maximal_reconvergence */
      .shaderMaximalReconvergence = true,

      /* VK_KHR_shader_relaxed_extended_instruction */
      .shaderRelaxedExtendedInstruction = true,

      /* VK_KHR_shader_untyped_pointers */
      .shaderUntypedPointers = true,

#ifdef KK_USE_WSI_PLATFORM
      /* VK_KHR_swapchain_maintenance1 */
      .swapchainMaintenance1 = true,
#endif

      /* VK_KHR_unified_image_layouts */
      .unifiedImageLayouts = true,
      .unifiedImageLayoutsVideo = false,

      /* VK_KHR_workgroup_memory_explicit_layout */
      .workgroupMemoryExplicitLayout = true,
      .workgroupMemoryExplicitLayoutScalarBlockLayout = true,
      .workgroupMemoryExplicitLayout8BitAccess = true,
      .workgroupMemoryExplicitLayout16BitAccess = true,

      /* EXT_4444_formats */
      .formatA4R4G4B4 = true,
      .formatA4B4G4R4 = true,

      /* VK_EXT_attachment_feedback_loop_layout */
      .attachmentFeedbackLoopLayout = true,

      /* VK_EXT_attachment_feedback_loop_dynamic_state */
      .attachmentFeedbackLoopDynamicState = true,

      /* VK_EXT_blend_operation_advanced */
      .advancedBlendCoherentOperations = true,

      /* VK_EXT_border_color_swizzle */
      .borderColorSwizzle = supported_extensions->EXT_border_color_swizzle,
      .borderColorSwizzleFromImage = false,

      /* VK_EXT_conditional_rendering */
      .conditionalRendering = true,
      .inheritedConditionalRendering = true,

      /* VK_EXT_custom_border_color */
      .customBorderColors = supported_extensions->EXT_custom_border_color,
      .customBorderColorWithoutFormat =
         supported_extensions->EXT_custom_border_color,

      /* VK_EXT_custom_resolve */
      .customResolve = true,

      /* VK_EXT_depth_clip_control */
      .depthClipControl = true,

      /* VK_EXT_depth_clip_enable */
      .depthClipEnable = true,

      /* VK_EXT_extended_dynamic_state3 */
      .extendedDynamicState3DepthClampEnable = true,
      .extendedDynamicState3DepthClipEnable = true,
      .extendedDynamicState3DepthClipNegativeOneToOne = true,
      .extendedDynamicState3LineRasterizationMode = true,
      .extendedDynamicState3ProvokingVertexMode = true,
      .extendedDynamicState3SampleLocationsEnable = true,
      .extendedDynamicState3TessellationDomainOrigin = true,

      /* EXT_image_2d_view_of_3d */
      .image2DViewOf3D = true,
      .sampler2DViewOf3D = true,

      /* VK_EXT_image_view_min_lod */
      .minLod = supported_extensions->EXT_image_view_min_lod,

      /* VK_EXT_map_memory_placed */
      .memoryMapPlaced = true,
      .memoryMapRangePlaced = false,
      .memoryUnmapReserve = true,

      /* VK_EXT_multi_draw */
      .multiDraw = true,

      /* VK_EXT_nested_command_buffer */
      .nestedCommandBuffer = true,
      .nestedCommandBufferRendering = true,
      .nestedCommandBufferSimultaneousUse = true,

      /* VK_EXT_primitive_restart_index */
      .primitiveRestartIndex = true,

      /* VK_EXT_primitive_topology_list_restart */
      .primitiveTopologyListRestart = true,
      .primitiveTopologyPatchListRestart = false,

      /* VK_EXT_provoking_vertex */
      .provokingVertexLast = true,

      /* VK_EXT_shader_replicated_composites */
      .shaderReplicatedComposites = true,

      /* VK_KHR_shader_subgroup_uniform_control_flow */
      .shaderSubgroupUniformControlFlow = true,

      /* VK_EXT_shader_atomic_float */
      .shaderBufferFloat32Atomics = true,
      .shaderBufferFloat32AtomicAdd = true,
      .shaderSharedFloat32Atomics = true,
      .shaderSharedFloat32AtomicAdd =
         pdev->info.msl_version >= MTL_LANGUAGE_VERSION_4_1,

      /* VK_EXT_vertex_attribute_robustness */
      .vertexAttributeRobustness = true,

      /* EXT_ycbcr_2plane_444_formats */
      .ycbcr2plane444Formats = true,
   };
}

static void
kk_get_device_properties(
   const struct kk_physical_device *pdev, const struct kk_instance *instance,
   const struct vk_device_extension_table *supported_extensions,
   struct vk_properties *properties)
{
   VkSampleCountFlags sample_counts = pdev->info.supported_sample_counts;

   uint64_t os_page_size = 4096;
   os_get_page_size(&os_page_size);

   // Queries the frequency of the GPU timestamp in ticks per second.
   uint64_t timestamp_frequency =
      mtl_device_timestamp_frequency(pdev->mtl_dev_handle);
   float timestamp_period =
      timestamp_frequency ? (1000000000.0f / (float)timestamp_frequency) : 1.0f;

   /* M1 and M2 are restricted to 1024 unique samplers, so there we advertise
    * the minimum allowed by spec. Otherwise, we advertise the maximum that fit
    * in our sampler table. */
   uint32_t max_samplers = pdev->info.max_sampler_count >= MSL_MAX_SAMPLERS
                              ? MSL_MAX_SAMPLERS
                              : 4000;

   *properties = (struct vk_properties){
      .apiVersion = kk_get_vk_version(),
      .driverVersion = vk_get_driver_version(),
      .vendorID =
         instance->force_vk_vendor != 0 ? instance->force_vk_vendor : 0x106b,
      .deviceID = 100,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,

      /* Vulkan 1.0 limits */
      /* Values taken from Apple7
         https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf */
      .maxImageDimension1D = kk_image_max_dimension(pdev, VK_IMAGE_TYPE_1D),
      .maxImageDimension2D = kk_image_max_dimension(pdev, VK_IMAGE_TYPE_2D),
      .maxImageDimension3D = kk_image_max_dimension(pdev, VK_IMAGE_TYPE_3D),
      .maxImageDimensionCube = kk_image_max_dimension(pdev, VK_IMAGE_TYPE_2D),
      .maxImageArrayLayers = 2048,
      .maxTexelBufferElements = 16384 * 16384,
      .maxUniformBufferRange = 65536,
      .maxStorageBufferRange = UINT32_MAX,
      .maxPushConstantsSize = KK_MAX_PUSH_SIZE,
      .maxMemoryAllocationCount = 4096,
      .maxSamplerAllocationCount = max_samplers,
      .bufferImageGranularity = 16,
      .sparseAddressSpaceSize = KK_SPARSE_ADDR_SPACE_SIZE,
      .maxBoundDescriptorSets = KK_MAX_SETS,
      .maxPerStageDescriptorSamplers = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUniformBuffers = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorStorageBuffers = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorSampledImages = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorStorageImages = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorInputAttachments = KK_MAX_DESCRIPTORS,
      .maxPerStageResources = UINT32_MAX,
      .maxDescriptorSetSamplers = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetUniformBuffers = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetUniformBuffersDynamic = KK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetStorageBuffers = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetStorageBuffersDynamic = KK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetSampledImages = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetStorageImages = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetInputAttachments = KK_MAX_DESCRIPTORS,
      .maxVertexInputAttributes = KK_MAX_ATTRIBS,
      .maxVertexInputBindings = KK_MAX_VBUFS,
      .maxVertexInputAttributeOffset = 2047,
      .maxVertexInputBindingStride = 2048,
      .maxVertexOutputComponents = 128,
      .maxTessellationGenerationLevel = 64,
      .maxTessellationPatchSize = 32,
      .maxTessellationControlPerVertexInputComponents = 128,
      .maxTessellationControlPerVertexOutputComponents = 128,
      .maxTessellationControlPerPatchOutputComponents = 120,
      .maxTessellationControlTotalOutputComponents = 4216,
      .maxTessellationEvaluationInputComponents = 128,
      .maxTessellationEvaluationOutputComponents = 128,
      .maxGeometryShaderInvocations = 32,
      .maxGeometryInputComponents = 128,
      .maxGeometryOutputComponents = 128,
      .maxGeometryOutputVertices = 1024,
      .maxGeometryTotalOutputComponents = 1024,
      .maxFragmentInputComponents = 128,
      .maxFragmentOutputAttachments = KK_MAX_RTS,
      .maxFragmentDualSrcAttachments = 1,
      .maxFragmentCombinedOutputResources = KK_MAX_DESCRIPTORS,
      .maxComputeSharedMemorySize = pdev->info.max_compute_shared_memory_size,
      .maxComputeWorkGroupCount = {0x7fffffff, 65535, 65535},
      .maxComputeWorkGroupInvocations = pdev->info.max_workgroup_invocations,
      .maxComputeWorkGroupSize = {pdev->info.max_workgroup_count[0],
                                  pdev->info.max_workgroup_count[1],
                                  pdev->info.max_workgroup_count[2]},
      .subPixelPrecisionBits = 8,
      .subTexelPrecisionBits = 8,
      .mipmapPrecisionBits = 8,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = UINT16_MAX,
      .maxSamplerLodBias = 15,
      .maxSamplerAnisotropy = 16,
      .maxViewports = KK_MAX_VIEWPORTS,
      .maxViewportDimensions = {32768, 32768},
      .viewportBoundsRange = {-65536, 65536},
      .viewportSubPixelBits = 8,
      .minMemoryMapAlignment = os_page_size,
      .minTexelBufferOffsetAlignment = KK_MIN_TEXEL_BUFFER_ALIGNMENT,
      .minUniformBufferOffsetAlignment = KK_MIN_UBO_ALIGNMENT,
      .minStorageBufferOffsetAlignment = KK_MIN_SSBO_ALIGNMENT,
      .minTexelOffset = -8,
      .maxTexelOffset = 7,
      .minTexelGatherOffset = -8,
      .maxTexelGatherOffset = 7,
      .minInterpolationOffset = -0.5,
      .maxInterpolationOffset = 0.4375,
      .subPixelInterpolationOffsetBits = 4,
      .maxFramebufferHeight = 16384,
      .maxFramebufferWidth = 16384,
      .maxFramebufferLayers = 2048,
      .framebufferColorSampleCounts = sample_counts,
      .framebufferDepthSampleCounts = sample_counts,
      .framebufferNoAttachmentsSampleCounts = sample_counts,
      .framebufferStencilSampleCounts = sample_counts,
      .maxColorAttachments = KK_MAX_RTS,
      .sampledImageColorSampleCounts = sample_counts,
      .sampledImageIntegerSampleCounts = sample_counts,
      .sampledImageDepthSampleCounts = sample_counts,
      .sampledImageStencilSampleCounts = sample_counts,
      .storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxSampleMaskWords = 1,
      .timestampComputeAndGraphics = true,
      .timestampPeriod = timestamp_period,
      .maxClipDistances = 8,
      .maxCullDistances = 8,
      .maxCombinedClipAndCullDistances = 8,
      .discreteQueuePriorities = 2,
      .pointSizeRange = {1.0f, 511.0f},
      .lineWidthRange = {1.0f, 1.0f},
      .pointSizeGranularity = 0.0625f,
      .lineWidthGranularity = 0.0f,
      .strictLines = false,
      .standardSampleLocations = true,
      .optimalBufferCopyOffsetAlignment = 1,
      .optimalBufferCopyRowPitchAlignment = 1,
      .nonCoherentAtomSize = 64,

      /* Vulkan 1.0 sparse properties */
      .sparseResidencyNonResidentStrict = false,
      .sparseResidencyAlignedMipSize = false,
      .sparseResidencyStandard2DBlockShape = false,
      .sparseResidencyStandard2DMultisampleBlockShape = false,
      .sparseResidencyStandard3DBlockShape = false,

      /* Vulkan 1.1 properties */
      .subgroupSize = 32,
      .subgroupSupportedStages =
         VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
         VK_SHADER_STAGE_VERTEX_BIT, /* kosmickrisp-vertex-subgroups */
      .subgroupSupportedOperations =
         VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
         VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
         VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
         VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
         VK_SUBGROUP_FEATURE_CLUSTERED_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT |
         VK_SUBGROUP_FEATURE_ROTATE_BIT |
         VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT,
      .subgroupQuadOperationsInAllStages = true,
      .pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_USER_CLIP_PLANES_ONLY,
      .maxMultiviewViewCount = KK_MAX_MULTIVIEW_VIEW_COUNT,
      .maxMultiviewInstanceIndex = UINT32_MAX,
      .maxPerSetDescriptors = UINT32_MAX,
      .maxMemoryAllocationSize = pdev->info.max_buffer_size,

      /* Vulkan 1.2 properties */
      .supportedDepthResolveModes =
         VK_RESOLVE_MODE_SAMPLE_ZERO_BIT | VK_RESOLVE_MODE_AVERAGE_BIT |
         VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT |
         VK_RESOLVE_MODE_CUSTOM_BIT_EXT,
      .supportedStencilResolveModes =
         VK_RESOLVE_MODE_SAMPLE_ZERO_BIT | VK_RESOLVE_MODE_MIN_BIT |
         VK_RESOLVE_MODE_MAX_BIT | VK_RESOLVE_MODE_CUSTOM_BIT_EXT,
      .independentResolveNone = true,
      .independentResolve = true,
      .driverID = VK_DRIVER_ID_MESA_KOSMICKRISP,
      .conformanceVersion = (VkConformanceVersion){1, 4, 6, 2},
      .denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE,
      .roundingModeIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE,
      .shaderSignedZeroInfNanPreserveFloat16 = true,
      .shaderSignedZeroInfNanPreserveFloat32 = true,
      .shaderSignedZeroInfNanPreserveFloat64 = false,
      .shaderDenormPreserveFloat16 = false,
      .shaderDenormPreserveFloat32 = false,
      .shaderDenormPreserveFloat64 = false,
      .shaderDenormFlushToZeroFloat16 = false,
      .shaderDenormFlushToZeroFloat32 = false,
      .shaderDenormFlushToZeroFloat64 = false,
      .shaderRoundingModeRTEFloat16 = true,
      .shaderRoundingModeRTEFloat32 = true,
      .shaderRoundingModeRTEFloat64 = false,
      .shaderRoundingModeRTZFloat16 = false,
      .shaderRoundingModeRTZFloat32 = false,
      .shaderRoundingModeRTZFloat64 = false,
      .maxUpdateAfterBindDescriptorsInAllPools = UINT32_MAX,
      .shaderUniformBufferArrayNonUniformIndexingNative = true,
      .shaderSampledImageArrayNonUniformIndexingNative = true,
      .shaderStorageBufferArrayNonUniformIndexingNative = true,
      .shaderStorageImageArrayNonUniformIndexingNative = true,
      .shaderInputAttachmentArrayNonUniformIndexingNative = true,
      .robustBufferAccessUpdateAfterBind = true,
      .quadDivergentImplicitLod = false,
      .maxPerStageDescriptorUpdateAfterBindSamplers = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindUniformBuffers = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindStorageBuffers = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindSampledImages = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindStorageImages = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindInputAttachments =
         KK_MAX_DESCRIPTORS,
      .maxPerStageUpdateAfterBindResources = UINT32_MAX,
      .maxDescriptorSetUpdateAfterBindSamplers = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindUniformBuffers = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindUniformBuffersDynamic =
         KK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindStorageBuffers = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindStorageBuffersDynamic =
         KK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindSampledImages = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindStorageImages = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindInputAttachments = KK_MAX_DESCRIPTORS,
      .filterMinmaxSingleComponentFormats =
         supported_extensions->EXT_sampler_filter_minmax,
      .filterMinmaxImageComponentMapping =
         supported_extensions->EXT_sampler_filter_minmax,
      .maxTimelineSemaphoreValueDifference = UINT64_MAX,
      .framebufferIntegerColorSampleCounts = sample_counts,

      /* VK_KHR_acceleration_structure: Vulkan's minimum limits. Metal's
       * default (not extended) limits cover them. */
      .maxGeometryCount = (1u << 24) - 1,
      .maxInstanceCount = (1u << 24) - 1,
      .maxPrimitiveCount = (1u << 29) - 1,
      .maxPerStageDescriptorAccelerationStructures = KK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindAccelerationStructures =
         KK_MAX_DESCRIPTORS,
      .maxDescriptorSetAccelerationStructures = KK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindAccelerationStructures =
         KK_MAX_DESCRIPTORS,
      .minAccelerationStructureScratchOffsetAlignment = KK_AS_SCRATCH_ALIGNMENT,

      /* VK_KHR_ray_tracing_pipeline. Metal reads 12 bits of the intersection
       * function buffer stride, which is the hit group stride. */
      .shaderGroupHandleSize = 32,
      .maxRayRecursionDepth = 31,
      .maxShaderGroupStride = 4064,
      .shaderGroupBaseAlignment = 64,
      .shaderGroupHandleCaptureReplaySize = 32,
      .maxRayDispatchInvocationCount = 1u << 30,
      .shaderGroupHandleAlignment = 32,
      .maxRayHitAttributeSize = 32,

      /* Vulkan 1.3 properties */
      .minSubgroupSize = 32,
      .maxSubgroupSize = 32,
      .maxComputeWorkgroupSubgroups = pdev->info.max_workgroup_invocations / 32,
      .requiredSubgroupSizeStages = 0,
      .maxInlineUniformBlockSize = KK_MAX_INLINE_UNIFORM_BLOCK_SIZE,
      .maxPerStageDescriptorInlineUniformBlocks = 32,
      .maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = 32,
      .maxDescriptorSetInlineUniformBlocks = 6 * 32,
      .maxDescriptorSetUpdateAfterBindInlineUniformBlocks = 6 * 32,
      .maxInlineUniformTotalSize = 1 << 16,
      .integerDotProduct4x8BitPackedUnsignedAccelerated = false,
      .integerDotProduct4x8BitPackedSignedAccelerated = false,
      .integerDotProduct4x8BitPackedMixedSignednessAccelerated = false,
      .storageTexelBufferOffsetAlignmentBytes = KK_MIN_TEXEL_BUFFER_ALIGNMENT,
      .storageTexelBufferOffsetSingleTexelAlignment = false,
      .uniformTexelBufferOffsetAlignmentBytes = KK_MIN_TEXEL_BUFFER_ALIGNMENT,
      .uniformTexelBufferOffsetSingleTexelAlignment = false,
      .maxBufferSize = pdev->info.max_buffer_size,

      /* VK_KHR_push_descriptor */
      .maxPushDescriptors = KK_MAX_PUSH_DESCRIPTORS,

      /* VK_EXT_blend_operation_advanced */
      .advancedBlendMaxColorAttachments = KK_MAX_RTS,
      .advancedBlendIndependentBlend = true,
      .advancedBlendNonPremultipliedSrcColor = true,
      .advancedBlendNonPremultipliedDstColor = true,
      .advancedBlendCorrelatedOverlap = true,
      .advancedBlendAllOperations = true,

      /* VK_EXT_custom_border_color */
      .maxCustomBorderColorSamplers = max_samplers,

      /* VK_EXT_extended_dynamic_state3 */
      .dynamicPrimitiveTopologyUnrestricted = false,

      /* VK_EXT_external_memory_host */
      .minImportedHostPointerAlignment = os_page_size,

      /* VK_EXT_graphics_pipeline_library */
      .graphicsPipelineLibraryFastLinking = true,
      .graphicsPipelineLibraryIndependentInterpolationDecoration = true,

      /* VK_KHR_line_rasterization */
      .lineSubPixelPrecisionBits = 8,

      /* VK_KHR_maintenance5 */
      .earlyFragmentMultisampleCoverageAfterSampleCounting = true,
      .earlyFragmentSampleMaskTestBeforeSampleCounting = false,
      .depthStencilSwizzleOneSupport = true,
      .polygonModePointSize = false,
      .nonStrictSinglePixelWideLinesUseParallelogram = false,
      .nonStrictWideLinesUseParallelogram = false,

      /* VK_KHR_maintenance6 */
      .blockTexelViewCompatibleMultipleLayers = false,
      .maxCombinedImageSamplerDescriptorCount = 3,
      .fragmentShadingRateClampCombinerInputs = false, /* TODO */

      /* VK_KHR_maintenance7 */
      .robustFragmentShadingRateAttachmentAccess = false,
      .separateDepthStencilAttachmentAccess = true,
      .maxDescriptorSetTotalUniformBuffersDynamic = KK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetTotalStorageBuffersDynamic = KK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetTotalBuffersDynamic = KK_MAX_DYNAMIC_BUFFERS,
      .maxDescriptorSetUpdateAfterBindTotalUniformBuffersDynamic =
         KK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindTotalStorageBuffersDynamic =
         KK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindTotalBuffersDynamic =
         KK_MAX_DYNAMIC_BUFFERS,

      /* VK_KHR_maintenance9 */
      .image2DViewOf3DSparse = false,
      .defaultVertexAttributeValue =
         VK_DEFAULT_VERTEX_ATTRIBUTE_VALUE_ZERO_ZERO_ZERO_ONE_KHR,

      /* VK_KHR_maintenance10 */
      .rgba4OpaqueBlackSwizzled = false,
      .resolveSrgbFormatAppliesTransferFunction = true,
      .resolveSrgbFormatSupportsTransferFunctionControl = true,

      /* VK_EXT_legacy_vertex_attributes */
      .nativeUnalignedPerformance = true,

      /* VK_EXT_map_memory_placed */
      .minPlacedMemoryMapAlignment = os_page_size,

      /* VK_EXT_multi_draw */
      .maxMultiDrawCount = UINT32_MAX,

      /* VK_EXT_nested_command_buffer */
      .maxCommandBufferNestingLevel = UINT32_MAX,

      /* VK_EXT_pipeline_robustness */
      .defaultRobustnessStorageBuffers =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .defaultRobustnessUniformBuffers =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .defaultRobustnessVertexInputs =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .defaultRobustnessImages =
         VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2,

      /* VK_EXT_physical_device_drm gets populated later */

      /* VK_EXT_provoking_vertex */
      .provokingVertexModePerPipeline = true,
      .transformFeedbackPreservesTriangleFanProvokingVertex = true,

      /* VK_KHR_robustness2 */
      .robustStorageBufferAccessSizeAlignment = KK_SSBO_BOUNDS_CHECK_ALIGNMENT,
      .robustUniformBufferAccessSizeAlignment = KK_MIN_UBO_ALIGNMENT,

      /* VK_EXT_sample_locations */
      /* Metal does not support sample positions for single sample */
      .sampleLocationSampleCounts = sample_counts & ~VK_SAMPLE_COUNT_1_BIT,
      .maxSampleLocationGridSize = (VkExtent2D){1, 1},
      .sampleLocationCoordinateRange[0] = KK_MIN_SAMPLE_LOCATION,
      .sampleLocationCoordinateRange[1] = KK_MAX_SAMPLE_LOCATION,
      .sampleLocationSubPixelBits = 4,
      .variableSampleLocations = true,

      /* VK_EXT_shader_object */
      .shaderBinaryVersion = 0,

      /* VK_EXT_transform_feedback */
      .maxTransformFeedbackStreams = 4,
      .maxTransformFeedbackBuffers = 4,
      .maxTransformFeedbackBufferSize = UINT32_MAX,
      .maxTransformFeedbackStreamDataSize = 2048,
      .maxTransformFeedbackBufferDataSize = 512,
      .maxTransformFeedbackBufferDataStride = 2048,
      .transformFeedbackQueries = true,
      .transformFeedbackStreamsLinesTriangles = false,
      .transformFeedbackRasterizationStreamSelect = true,
      .transformFeedbackDraw = true,

      /* VK_KHR_vertex_attribute_divisor */
      .maxVertexAttribDivisor = UINT32_MAX,
      .supportsNonZeroFirstInstance = true,

      /* VK_KHR_fragment_shader_barycentric */
      .triStripVertexOrderIndependentOfProvokingVertex = false,
   };

   char gpu_name[256u];
   mtl_device_get_name(pdev->mtl_dev_handle, gpu_name);
   snprintf(properties->deviceName, sizeof(properties->deviceName), "%s",
            gpu_name);

   /* Not sure if there are layout specific things, so for now just reporting
    * all layouts from extensions.
    */
   static const VkImageLayout supported_layouts[] = {
      VK_IMAGE_LAYOUT_GENERAL, /* this one is required by spec */
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PREINITIALIZED,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
      VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
      VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ,
   };

   properties->pCopySrcLayouts = (VkImageLayout *)supported_layouts;
   properties->copySrcLayoutCount = ARRAY_SIZE(supported_layouts);
   properties->pCopyDstLayouts = (VkImageLayout *)supported_layouts;
   properties->copyDstLayoutCount = ARRAY_SIZE(supported_layouts);

   STATIC_ASSERT(sizeof(instance->driver_build_sha) >= VK_UUID_SIZE);
   memcpy(properties->optimalTilingLayoutUUID, instance->driver_build_sha,
          VK_UUID_SIZE);

   /* We're a UMR so we can always map every kind of memory */
   properties->identicalMemoryTypeRequirements = true;

   /* VK_EXT_shader_module_identifier */
   STATIC_ASSERT(sizeof(vk_shaderModuleIdentifierAlgorithmUUID) ==
                 sizeof(properties->shaderModuleIdentifierAlgorithmUUID));
   memcpy(properties->shaderModuleIdentifierAlgorithmUUID,
          vk_shaderModuleIdentifierAlgorithmUUID,
          sizeof(properties->shaderModuleIdentifierAlgorithmUUID));

   const struct {
      uint64_t registry_id;
      uint64_t pad;
   } dev_uuid = {
      .registry_id = mtl_device_get_registry_id(pdev->mtl_dev_handle),
   };
   STATIC_ASSERT(sizeof(dev_uuid) == VK_UUID_SIZE);
   memcpy(properties->deviceUUID, &dev_uuid, VK_UUID_SIZE);
   STATIC_ASSERT(sizeof(instance->driver_build_sha) >= VK_UUID_SIZE);
   memcpy(properties->driverUUID, instance->driver_build_sha, VK_UUID_SIZE);

   /* kosmickrisp-metalfx: NGX finds its adapter through the device LUID and
    * gives up without one. The Metal registry ID is unique per GPU. */
   if (pdev->settings.nvx_enabled) {
      STATIC_ASSERT(sizeof(dev_uuid.registry_id) == VK_LUID_SIZE);
      memcpy(properties->deviceLUID, &dev_uuid.registry_id, VK_LUID_SIZE);
      properties->deviceLUIDValid = true;
      properties->deviceNodeMask = 1;
   }

   snprintf(properties->driverName, VK_MAX_DRIVER_NAME_SIZE, "KosmicKrisp");
   snprintf(properties->driverInfo, VK_MAX_DRIVER_INFO_SIZE,
            "Mesa " PACKAGE_VERSION MESA_GIT_SHA1);
   kk_apply_vendor_identity(properties);
   kk_apply_driver_identity(properties);

   if (kk_log_queries_enabled()) {
      fprintf(stderr,
              "[kk] identity: vendorID 0x%04x deviceID 0x%04x deviceName \"%s\" "
              "driverID %d (%s) driverName \"%s\" driverInfo \"%s\" "
              "driverVersion 0x%08x\n",
              properties->vendorID, properties->deviceID, properties->deviceName,
              (int)properties->driverID,
              vk_DriverId_to_str(properties->driverID), properties->driverName,
              properties->driverInfo, properties->driverVersion);
   }
}

/* Settings that change shader compilation, for the cache keys */
static uint64_t
kk_physical_device_compiler_flags(const struct kk_physical_device *pdev)
{
   return (uint64_t)pdev->settings.experimental_enabled |
          (uint64_t)pdev->settings.ray_tracing_enabled << 1 |
          (uint64_t)pdev->settings.sparse_enabled << 2 |
          (uint64_t)pdev->settings.nvx_enabled << 3 |
          (uint64_t)pdev->settings.exact_output << 4;
}

static void
kk_physical_device_init_pipeline_cache(struct kk_physical_device *pdev)
{
   struct kk_instance *instance = kk_physical_device_instance(pdev);

   blake3_hasher blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   _mesa_blake3_update(&blake3_ctx, instance->driver_build_sha,
                       sizeof(instance->driver_build_sha));
   /* Settings that change the shaders we generate: a cache made under other
    * ones must not be taken */
   const uint64_t flags = kk_physical_device_compiler_flags(pdev);
   _mesa_blake3_update(&blake3_ctx, &flags, sizeof(flags));
   _mesa_blake3_update(&blake3_ctx, &pdev->settings.disabled_workarounds,
                       sizeof(pdev->settings.disabled_workarounds));

   unsigned char blake3[BLAKE3_KEY_LEN];
   _mesa_blake3_final(&blake3_ctx, blake3);

   STATIC_ASSERT(BLAKE3_KEY_LEN >= VK_UUID_SIZE);
   memcpy(pdev->vk.properties.pipelineCacheUUID, blake3, VK_UUID_SIZE);
   memcpy(pdev->vk.properties.shaderBinaryUUID, blake3, VK_UUID_SIZE);

#ifdef ENABLE_SHADER_CACHE
   /* Compiled shaders (NIR to MSL) are kept across runs, so a game's
    * pipelines don't go through SPIR-V, NIR and MSL generation again every
    * launch; Metal caches its own compiles of the MSL */
   char timestamp[BLAKE3_HEX_LEN];
   _mesa_blake3_format(timestamp, instance->driver_build_sha);
   pdev->vk.disk_cache =
      disk_cache_create("kosmickrisp", timestamp,
                        flags ^ pdev->settings.disabled_workarounds);
#endif
}

static void
kk_physical_device_free_disk_cache(struct kk_physical_device *pdev)
{
#ifdef ENABLE_SHADER_CACHE
   if (pdev->vk.disk_cache) {
      disk_cache_destroy(pdev->vk.disk_cache);
      pdev->vk.disk_cache = NULL;
   }
#else
   assert(pdev->vk.disk_cache == NULL);
#endif
}

static uint64_t
kk_get_sysmem_heap_size(struct kk_physical_device *pdev)
{
   /* Report the recommended Metal working set size as the GPU heap size. This
    * is a fixed percent of the total available system memory. */
   return mtl_device_recommended_max_working_set_size(pdev->mtl_dev_handle);
}

static uint64_t
kk_get_sysmem_heap_used(struct kk_physical_device *pdev)
{
   /* From the Vulkan 1.3.278 spec:
    *
    *    "heapUsage is an array of VK_MAX_MEMORY_HEAPS VkDeviceSize
    *    values in which memory usages are returned, with one element
    *    for each memory heap. A heap’s usage is an estimate of how
    *    much memory the process is currently using in that heap."
    *
    * From Metal documentation for currentAllocatedSize:
    *
    *     The total amount of memory, in bytes, the GPU device is using for all
    *     of its resources.
    *
    * We can trivially report estimated heap usage using Metal's reported
    * allocated size
    */
   return mtl_device_current_allocated_size(pdev->mtl_dev_handle);
}

static uint64_t
kk_get_sysmem_heap_budget(struct kk_physical_device *pdev)
{
   uint64_t heap_size = kk_get_sysmem_heap_size(pdev);
   uint64_t used = kk_get_sysmem_heap_used(pdev);

   /* Budget is calculated using the default Mesa logic, based on available
    * system memory. Available memory is reduced to 90% to avoid thrashing. */
   const float available_percent = 0.9f;
   return vk_physical_device_heap_budget_from_system(
      &pdev->vk, available_percent, heap_size, used);
}

static void
get_metal_limits(struct kk_physical_device *pdev)
{
   struct mtl_size workgroup_size =
      mtl_device_max_threads_per_threadgroup(pdev->mtl_dev_handle);
   pdev->info.max_workgroup_count[0] = workgroup_size.x;
   pdev->info.max_workgroup_count[1] = workgroup_size.y;
   pdev->info.max_workgroup_count[2] = workgroup_size.z;
   pdev->info.max_workgroup_invocations =
      MAX3(workgroup_size.x, workgroup_size.y, workgroup_size.z);

   pdev->info.max_compute_shared_memory_size =
      mtl_device_max_threadgroup_memory_length(pdev->mtl_dev_handle);
   pdev->info.max_buffer_size =
      mtl_device_max_buffer_length(pdev->mtl_dev_handle);
   pdev->info.max_sampler_count =
      mtl_device_max_argument_buffer_sampler_count(pdev->mtl_dev_handle);
   pdev->info.gpu_apple_family =
      mtl_device_get_gpu_apple_family(pdev->mtl_dev_handle);

   /* Determine the supported MSL version based on the OS. The version used to
    * compile determines what features are available. */
   if (ns_is_os_version_at_least(27, 0, 0))
      pdev->info.msl_version = MTL_LANGUAGE_VERSION_4_1;
   else
      pdev->info.msl_version = MTL_LANGUAGE_VERSION_4_0;

   /* See Metal Feature Set Tables. Note that for certain MSAA sample counts the
    * tile size will actually be restricted to a width and/or height of 16, but
    * we typically don't know the actual sample count when querying granularity
    * or checking render area alignment. Use 32x32 always as a best effort
    * optimal rendering area, which will also ensure proper alignment for 16
    * wide/tall tiles chosen by Metal. */
   pdev->info.rendering_tile_width = 32;
   pdev->info.rendering_tile_height = 32;

   pdev->info.supported_sample_counts = VK_SAMPLE_COUNT_1_BIT;
   for (uint32_t sample_count = VK_SAMPLE_COUNT_2_BIT;
        sample_count <= VK_SAMPLE_COUNT_8_BIT; sample_count <<= 1) {
      if (mtl_device_supports_sample_count(pdev->mtl_dev_handle, sample_count))
         pdev->info.supported_sample_counts |= sample_count;
   }
   assert(pdev->info.supported_sample_counts <= (KK_MAX_SAMPLES << 1) - 1);
}

static void
kk_parse_environment_options(struct kk_physical_device *pdev)
{
   struct kk_env_settings *settings = &pdev->settings;

   settings->gpu_capture_enabled =
      debug_get_bool_option("MESA_KK_GPU_CAPTURE", false);
   settings->ray_tracing_enabled =
      debug_get_bool_option("KK_RAY_TRACING", false);
   settings->sparse_enabled = debug_get_bool_option("KK_SPARSE", false);
   settings->nvx_enabled = debug_get_bool_option("KK_NVX", false);
   settings->exact_output = debug_get_bool_option("KK_EXACT", false);
   settings->queue_count =
      CLAMP(debug_get_num_option("KK_QUEUE_COUNT", 1), 1, KK_MAX_MAIN_QUEUES);
   const char *compute = debug_get_option("KK_COMPUTE_QUEUES", "");
   for (uint32_t i = 0; i < KK_MAX_COMPUTE_FAMILIES && *compute; i++) {
      char *end;
      const long n = strtol(compute, &end, 10);
      settings->compute_queue_counts[i] = CLAMP(n, 0, KK_MAX_COMPUTE_QUEUES);
      compute = *end == ',' ? end + 1 : end + strlen(end);
   }
   settings->device_local_type =
      debug_get_bool_option("KK_DEVICE_LOCAL_TYPE", true);
   settings->private_device_local =
      debug_get_bool_option("KK_PRIVATE_DEVICE_LOCAL", true);

   const char *list = debug_get_option("MESA_KK_DISABLE_WORKAROUNDS", "");
   const char *all_workarounds = "all";
   const size_t all_len = strlen(all_workarounds);
   for (unsigned n; n = strcspn(list, ","), *list; list += MAX2(1, n)) {
      if (n == all_len && !strncmp(list, all_workarounds, n)) {
         settings->disabled_workarounds = UINT64_MAX;
         break;
      }

      int index = atoi(list);
      settings->disabled_workarounds |= BITFIELD64_BIT(index);
   }

   /* Workarounds resolved on macOS 27 */
   if (ns_is_os_version_at_least(27, 0, 0)) {
      /* 1-6 */
      settings->disabled_workarounds |= BITFIELD64_MASK(7);

      settings->disabled_workarounds |= BITFIELD64_BIT(8);
      settings->disabled_workarounds |= BITFIELD64_BIT(11);
      settings->disabled_workarounds |= BITFIELD64_BIT(12);
      settings->disabled_workarounds |= BITFIELD64_BIT(17);
      settings->disabled_workarounds |= BITFIELD64_BIT(18);
   }
   /* M5 only workarounds */
   if (pdev->info.gpu_apple_family >= 10) {
      settings->disabled_workarounds &= ~BITFIELD64_BIT(2);
   }
   if (pdev->info.gpu_apple_family < 10) {
      settings->disabled_workarounds |= BITFIELD64_BIT(16);
   }
   /* M3+ only workarounds */
   if (pdev->info.gpu_apple_family < 9) {
      settings->disabled_workarounds |= BITFIELD64_BIT(18);
   }
}

VkResult
kk_enumerate_physical_devices(struct vk_instance *_instance)
{
   struct kk_instance *instance = (struct kk_instance *)_instance;
   VkResult result;

   struct kk_physical_device *pdev =
      vk_zalloc(&instance->vk.alloc, sizeof(*pdev), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);

   if (pdev == NULL) {
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   pdev->mtl_dev_handle = mtl_device_create();
   if (!pdev->mtl_dev_handle) {
      result = VK_SUCCESS;
      goto fail_alloc;
   }
   get_metal_limits(pdev);
   kk_parse_environment_options(pdev);

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &kk_physical_device_entrypoints, true);
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_physical_device_entrypoints, false);

   struct vk_device_extension_table supported_extensions;
   kk_get_device_extensions(instance, pdev, &supported_extensions);

   struct vk_features supported_features;
   kk_get_device_features(pdev, &supported_extensions, &supported_features);

   struct vk_properties properties;
   kk_get_device_properties(pdev, instance, &supported_extensions, &properties);

   properties.drmHasRender = false;

   result = vk_physical_device_init(&pdev->vk, &instance->vk,
                                    &supported_extensions, &supported_features,
                                    &properties, &dispatch_table);
   if (result != VK_SUCCESS)
      goto fail_mtl_dev;

   kk_physical_device_init_pipeline_cache(pdev);

   uint64_t sysmem_size_B = kk_get_sysmem_heap_size(pdev);
   if (sysmem_size_B == 0) {
      result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to query total system memory");
      goto fail_disk_cache;
   }

   uint32_t sysmem_heap_idx = pdev->mem_heap_count++;
   pdev->mem_heaps[sysmem_heap_idx] = (struct kk_memory_heap){
      .size = sysmem_size_B,
      .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
      .budget = kk_get_sysmem_heap_budget,
      .used = kk_get_sysmem_heap_used,
   };

   /* Private memory, which isn't host visible: Metal places sparse resources
    * and acceleration structures only in private heaps. Only sparse resources
    * and acceleration structure storage buffers list this type. It comes
    * first, as the spec orders types whose flags are a subset of another's,
    * which also makes it the one allocators looking for the first
    * DEVICE_LOCAL type pick. Without either feature it doesn't exist and
    * private_mem_type is an index past the last type. */
   const bool private_mem =
      pdev->settings.sparse_enabled || pdev->settings.ray_tracing_enabled;
   if (private_mem) {
      pdev->private_mem_type = pdev->mem_type_count;
      pdev->mem_types[pdev->mem_type_count++] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
         .heapIndex = sysmem_heap_idx,
      };
   }

   /* A regular DEVICE_LOCAL type that isn't host visible. On unified memory
    * it's the same shared memory as the next type, but allocators that want
    * device memory the host can't see (AMD's FidelityFX runtime refuses
    * host-visible types for it and fails context creation without one) find
    * it. It sits before the host visible type, as its flags are a subset. */
   pdev->device_local_mem_type = UINT8_MAX;
   if (pdev->settings.device_local_type) {
      pdev->device_local_mem_type = pdev->mem_type_count;
      pdev->mem_types[pdev->mem_type_count++] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
         .heapIndex = sysmem_heap_idx,
      };
   }

   pdev->mem_types[pdev->mem_type_count++] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      .heapIndex = sysmem_heap_idx,
   };
   if (!private_mem)
      pdev->private_mem_type = pdev->mem_type_count;

   assert(pdev->mem_heap_count <= ARRAY_SIZE(pdev->mem_heaps));
   assert(pdev->mem_type_count <= ARRAY_SIZE(pdev->mem_types));

   pdev->queue_families[pdev->queue_family_count++] = (struct kk_queue_family){
      .queue_flags =
         VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT |
         (pdev->settings.sparse_enabled ? VK_QUEUE_SPARSE_BINDING_BIT : 0),
      .queue_count = pdev->settings.queue_count,
   };
   for (uint32_t i = 0; i < KK_MAX_COMPUTE_FAMILIES; i++) {
      if (!pdev->settings.compute_queue_counts[i])
         break;
      pdev->queue_families[pdev->queue_family_count++] = (struct kk_queue_family){
         .queue_flags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
         .queue_count = pdev->settings.compute_queue_counts[i],
      };
   }
   assert(pdev->queue_family_count <= ARRAY_SIZE(pdev->queue_families));

   pdev->sync_binary_type = vk_sync_binary_get_type(&kk_sync_type);
   unsigned st_idx = 0;
   pdev->sync_types[st_idx++] = &kk_sync_type;
   pdev->sync_types[st_idx++] = &pdev->sync_binary_type.sync;
   pdev->sync_types[st_idx++] = NULL;
   assert(st_idx <= ARRAY_SIZE(pdev->sync_types));
   pdev->vk.supported_sync_types = pdev->sync_types;

   result = kk_init_wsi(pdev);
   if (result != VK_SUCCESS)
      goto fail_disk_cache;

   simple_mtx_init(&pdev->softfp64_mutex, mtx_plain);
   list_add(&pdev->vk.link, &instance->vk.physical_devices.list);

   return VK_SUCCESS;

fail_disk_cache:
   vk_physical_device_finish(&pdev->vk);
fail_mtl_dev:
   mtl_release(pdev->mtl_dev_handle);
fail_alloc:
   vk_free(&instance->vk.alloc, pdev);
   return result;
}

void
kk_physical_device_destroy(struct vk_physical_device *vk_pdev)
{
   struct kk_physical_device *pdev =
      container_of(vk_pdev, struct kk_physical_device, vk);

   kk_finish_wsi(pdev);
   kk_physical_device_free_disk_cache(pdev);
   ralloc_free(pdev->softfp64);
   simple_mtx_destroy(&pdev->softfp64_mutex);
   vk_physical_device_finish(&pdev->vk);
   mtl_release(pdev->mtl_dev_handle);
   vk_free(&pdev->vk.instance->alloc, pdev);
}

static void
kk_get_physical_device_properties2(VkPhysicalDevice physicalDevice,
                                   VkPhysicalDeviceProperties2 *pProperties);

VKAPI_ATTR void VKAPI_CALL
kk_GetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice,
                                VkPhysicalDeviceProperties2 *pProperties)
{
   if (kk_log_queries_enabled())
      kk_log_chain_types("vkGetPhysicalDeviceProperties2", pProperties->pNext);

   kk_get_physical_device_properties2(physicalDevice, pProperties);

   if (kk_log_queries_enabled()) {
      kk_log_properties("vkGetPhysicalDeviceProperties2.properties",
                        &pProperties->properties);
      kk_log_chain("vkGetPhysicalDeviceProperties2", pProperties->pNext);
   }
}

/* VkPhysicalDeviceFeatures members, in order; the struct is only VkBool32s */
static const char *const kk_core_feature_names[] = {
   "robustBufferAccess",
   "fullDrawIndexUint32",
   "imageCubeArray",
   "independentBlend",
   "geometryShader",
   "tessellationShader",
   "sampleRateShading",
   "dualSrcBlend",
   "logicOp",
   "multiDrawIndirect",
   "drawIndirectFirstInstance",
   "depthClamp",
   "depthBiasClamp",
   "fillModeNonSolid",
   "depthBounds",
   "wideLines",
   "largePoints",
   "alphaToOne",
   "multiViewport",
   "samplerAnisotropy",
   "textureCompressionETC2",
   "textureCompressionASTC_LDR",
   "textureCompressionBC",
   "occlusionQueryPrecise",
   "pipelineStatisticsQuery",
   "vertexPipelineStoresAndAtomics",
   "fragmentStoresAndAtomics",
   "shaderTessellationAndGeometryPointSize",
   "shaderImageGatherExtended",
   "shaderStorageImageExtendedFormats",
   "shaderStorageImageMultisample",
   "shaderStorageImageReadWithoutFormat",
   "shaderStorageImageWriteWithoutFormat",
   "shaderUniformBufferArrayDynamicIndexing",
   "shaderSampledImageArrayDynamicIndexing",
   "shaderStorageBufferArrayDynamicIndexing",
   "shaderStorageImageArrayDynamicIndexing",
   "shaderClipDistance",
   "shaderCullDistance",
   "shaderFloat64",
   "shaderInt64",
   "shaderInt16",
   "shaderResourceResidency",
   "shaderResourceMinLod",
   "sparseBinding",
   "sparseResidencyBuffer",
   "sparseResidencyImage2D",
   "sparseResidencyImage3D",
   "sparseResidency2Samples",
   "sparseResidency4Samples",
   "sparseResidency8Samples",
   "sparseResidency16Samples",
   "sparseResidencyAliased",
   "variableMultisampleRate",
   "inheritedQueries",
};
static_assert(ARRAY_SIZE(kk_core_feature_names) * sizeof(VkBool32) ==
                 sizeof(VkPhysicalDeviceFeatures),
              "kk_core_feature_names covers VkPhysicalDeviceFeatures");

/* kosmickrisp-compat-switches: KK_REPORT_FEATURES=<comma list>|all reports those core
 * VkPhysicalDeviceFeatures as supported, for games that refuse to start without them.
 * Capabilities don't change: vkCreateDevice drops enabled features the GPU lacks (vk_device.c). */
static void
kk_apply_reported_features(VkPhysicalDeviceFeatures *features)
{
   const char *value = getenv("KK_REPORT_FEATURES");
   if (!value || !value[0])
      return;

   VkBool32 *fields = (VkBool32 *)features;
   if (strcmp(value, "all") == 0) {
      for (unsigned i = 0; i < ARRAY_SIZE(kk_core_feature_names); i++)
         fields[i] = VK_TRUE;
      return;
   }

   static bool warned = false;
   char list[1024];
   snprintf(list, sizeof(list), "%s", value);
   for (char *saved = NULL, *name = strtok_r(list, ", ", &saved); name;
        name = strtok_r(NULL, ", ", &saved)) {
      unsigned i = 0;
      while (i < ARRAY_SIZE(kk_core_feature_names) &&
             strcmp(name, kk_core_feature_names[i]) != 0)
         i++;
      if (i < ARRAY_SIZE(kk_core_feature_names))
         fields[i] = VK_TRUE;
      else if (!warned)
         fprintf(stderr, "[kk] KK_REPORT_FEATURES: unknown feature \"%s\"\n", name);
   }
   warned = true;
}

VKAPI_ATTR void VKAPI_CALL
kk_GetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice,
                              VkPhysicalDeviceFeatures2 *pFeatures)
{
   if (kk_log_queries_enabled())
      kk_log_chain_types("vkGetPhysicalDeviceFeatures2", pFeatures->pNext);

   vk_common_GetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
   kk_apply_reported_features(&pFeatures->features);

   if (kk_log_queries_enabled()) {
      kk_log_features("vkGetPhysicalDeviceFeatures2.features",
                      &pFeatures->features);
      kk_log_chain("vkGetPhysicalDeviceFeatures2", pFeatures->pNext);
   }
}

static void
kk_get_physical_device_properties2(VkPhysicalDevice physicalDevice,
                                   VkPhysicalDeviceProperties2 *pProperties)
{
   vk_common_GetPhysicalDeviceProperties2(physicalDevice, pProperties);

   /* Properly populate layered API properties */
   VkPhysicalDeviceLayeredApiPropertiesListKHR *layered_props_list =
      vk_find_struct(pProperties->pNext,
                     PHYSICAL_DEVICE_LAYERED_API_PROPERTIES_LIST_KHR);
   if (!layered_props_list)
      return;

   layered_props_list->layeredApiCount = 1;
   if (!layered_props_list->pLayeredApis)
      return;

   VkPhysicalDeviceLayeredApiPropertiesKHR *layered_props =
      &layered_props_list->pLayeredApis[0];
   layered_props->vendorID = pProperties->properties.vendorID;
   layered_props->deviceID = pProperties->properties.deviceID;
   layered_props->layeredAPI = VK_PHYSICAL_DEVICE_LAYERED_API_METAL_KHR;
   strncpy(layered_props->deviceName, pProperties->properties.deviceName,
           VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
}

VKAPI_ATTR void VKAPI_CALL
kk_GetPhysicalDeviceMemoryProperties2(
   VkPhysicalDevice physicalDevice,
   VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);

   pMemoryProperties->memoryProperties.memoryHeapCount = pdev->mem_heap_count;
   for (int i = 0; i < pdev->mem_heap_count; i++) {
      pMemoryProperties->memoryProperties.memoryHeaps[i] = (VkMemoryHeap){
         .size = pdev->mem_heaps[i].size,
         .flags = pdev->mem_heaps[i].flags,
      };
   }

   pMemoryProperties->memoryProperties.memoryTypeCount = pdev->mem_type_count;
   for (int i = 0; i < pdev->mem_type_count; i++) {
      pMemoryProperties->memoryProperties.memoryTypes[i] = pdev->mem_types[i];
   }

   vk_foreach_struct(sType, ext, pMemoryProperties->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT: {
         VkPhysicalDeviceMemoryBudgetPropertiesEXT *p = ext;

         for (unsigned i = 0; i < pdev->mem_heap_count; i++) {
            const struct kk_memory_heap *heap = &pdev->mem_heaps[i];
            p->heapBudget[i] = heap->budget ? heap->budget(pdev) : 0;
            p->heapUsage[i] = heap->used ? heap->used(pdev) : 0;
         }

         /* From the Vulkan 1.3.278 spec:
          *
          *    "The heapBudget and heapUsage values must be zero for array
          *    elements greater than or equal to
          *    VkPhysicalDeviceMemoryProperties::memoryHeapCount. The
          *    heapBudget value must be non-zero for array elements less than
          *    VkPhysicalDeviceMemoryProperties::memoryHeapCount."
          */
         for (unsigned i = pdev->mem_heap_count; i < VK_MAX_MEMORY_HEAPS; i++) {
            p->heapBudget[i] = 0u;
            p->heapUsage[i] = 0u;
         }
         break;
      }
      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }

   if (kk_log_queries_enabled()) {
      const VkPhysicalDeviceMemoryProperties *mem =
         &pMemoryProperties->memoryProperties;
      for (uint32_t i = 0; i < mem->memoryHeapCount; i++)
         fprintf(stderr,
                 "[kk] vkGetPhysicalDeviceMemoryProperties heap %u: size %" PRIu64
                 " flags 0x%x\n",
                 i, (uint64_t)mem->memoryHeaps[i].size, mem->memoryHeaps[i].flags);
      for (uint32_t i = 0; i < mem->memoryTypeCount; i++)
         fprintf(stderr,
                 "[kk] vkGetPhysicalDeviceMemoryProperties type %u: heap %u "
                 "flags 0x%x\n",
                 i, mem->memoryTypes[i].heapIndex,
                 mem->memoryTypes[i].propertyFlags);
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_GetPhysicalDeviceQueueFamilyProperties2(
   VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount,
   VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, pQueueFamilyProperties,
                          pQueueFamilyPropertyCount);

   for (uint8_t i = 0; i < pdev->queue_family_count; i++) {
      const struct kk_queue_family *queue_family = &pdev->queue_families[i];

      vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p)
      {
         p->queueFamilyProperties.queueFlags = queue_family->queue_flags;
         p->queueFamilyProperties.queueCount = queue_family->queue_count;
         p->queueFamilyProperties.timestampValidBits = 64;
         p->queueFamilyProperties.minImageTransferGranularity =
            (VkExtent3D){1, 1, 1};

         vk_foreach_struct(sType, ext, p->pNext) {
            switch (sType) {
            case VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES: {
               VkQueueFamilyGlobalPriorityProperties *pSub = ext;
               pSub->priorityCount = 1;
               pSub->priorities[0] = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;
               break;
            }

            case VK_STRUCTURE_TYPE_QUEUE_FAMILY_OPTIMAL_IMAGE_TRANSFER_GRANULARITY_PROPERTIES_KHR: {
               VkQueueFamilyOptimalImageTransferGranularityPropertiesKHR *pSub =
                  ext;
               pSub->optimalImageTransferGranularity =
                  p->queueFamilyProperties.minImageTransferGranularity;
               break;
            }

            default:
               vk_debug_ignored_stype(sType);
               break;
            }
         }
      }
   }

   if (kk_log_queries_enabled() && pQueueFamilyProperties) {
      for (uint32_t i = 0; i < *pQueueFamilyPropertyCount; i++) {
         const VkQueueFamilyProperties *p =
            &pQueueFamilyProperties[i].queueFamilyProperties;
         fprintf(stderr,
                 "[kk] vkGetPhysicalDeviceQueueFamilyProperties family %u: "
                 "flags 0x%x queueCount %u\n",
                 i, p->queueFlags, p->queueCount);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_GetPhysicalDeviceMultisamplePropertiesEXT(
   VkPhysicalDevice physicalDevice, VkSampleCountFlagBits samples,
   VkMultisamplePropertiesEXT *pMultisampleProperties)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);

   if (samples & pdev->vk.properties.sampleLocationSampleCounts) {
      pMultisampleProperties->maxSampleLocationGridSize = (VkExtent2D){1, 1};
   } else {
      pMultisampleProperties->maxSampleLocationGridSize = (VkExtent2D){0, 0};
   }
}
