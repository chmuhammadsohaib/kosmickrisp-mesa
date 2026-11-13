# KosmicKrisp with ray tracing, sparse memory and MetalFX

A fork of [Mesa](https://gitlab.freedesktop.org/mesa/mesa)'s KosmicKrisp, the Vulkan driver that runs on Apple's
Metal, aimed at running modern Vulkan games on Apple silicon Macs (through Wine/CrossOver). The first commit imports
Mesa 26.3.0 as it is in the official repository at
[`34ad0a45`](https://gitlab.freedesktop.org/mesa/mesa/-/commit/34ad0a45aab3d89e30166c36c141cd1142aff0d1) (26.3
branch); the commits after it are the changes below. Everything outside `src/kosmickrisp` and the Metal WSI is
untouched. Mesa's own readme is in [README.rst](README.rst).

Most additions are opt-in with environment variables, so with none set the driver behaves like upstream 26.3.

## Changes

| Commit | What it adds | Switch |
|---|---|---|
| compat changes for idtech games | Subgroup values in vertex shaders, NaN-safe min/max/saturate under Metal's fast math, a per-shader safe-math mode, MSL dumps | `KK_SAFE_MATH`, `KK_SAFE_MATH_LOG`, `KK_DUMP_MSL` |
| ray tracing | `VK_KHR_acceleration_structure`, `VK_KHR_ray_query` and `VK_KHR_ray_tracing_pipeline` on Metal's intersectors; ray tracing stages become Metal functions | `KK_RAY_TRACING=1` |
| sparse buffers and 2d images | `sparseBinding`, `sparseResidencyBuffer` and `sparseResidencyImage2D` on Metal sparse heaps | `KK_SPARSE=1` |
| debug and identity switches | Log device queries; report another vendor/driver identity or core features and extensions that games insist on (capabilities don't change) | `KK_LOG_QUERIES`, `KK_VENDOR_ID`, `KK_DRIVER_ID`, `KK_REPORT_FEATURES`, `KK_REPORT_EXTENSIONS` |
| rt pipeline libraries, host as commands, stats and fixes | Pipeline libraries, host-side acceleration structure commands, refit/rebuild of stale BLASes, pooled ray tracing stacks, statistics, shader record fixes | `KK_RT_STATS`, `KK_RT_REFIT_LIMIT`, `KK_RT_REFIT_MIN_PRIMS` |
| lower fp64 to software | 64-bit floats in shaders (Metal has none), lowered to software | |
| metalfx upscaling over VK_NVX_binary_import | A private channel over `VK_NVX_binary_import` and `VK_NVX_image_view_handle` that runs MetalFX's temporal upscaler (with sharpening and dynamic resolution); a device LUID. A DLSS or FSR runtime on the Windows side drives it, see `src/kosmickrisp/vulkan/kk_mfx_channel.h` | `KK_NVX=1`, `KK_MFX_SIGNS`, `KK_MFX_CONTENT_MODE`, `KK_MFX_CONTENT_SCALES` |
| msl translation fixes | SSA values declared in the scope that defines them, compressed MSL kept for compiled shaders, `shader_clock` | |
| KK_SYNC_DEBUG reports stalled waits | Says what the queues have done whenever a CPU wait stalls | `KK_SYNC_DEBUG=1` |
| metalfx frame interpolation | MetalFX's frame interpolator over the same channel (frame generation) | `KK_NVX=1` |
| KK_QUEUE_COUNT for more queues | Up to four queues in the main queue family | `KK_QUEUE_COUNT=n` |
| always enable the nvx extensions under KK_NVX | The channel's extensions on every device, for games that don't ask for them | `KK_NVX=1` |
| device local memory type that isn't host visible | A regular `DEVICE_LOCAL`-only memory type, which AMD's FidelityFX runtime needs | `KK_DEVICE_LOCAL_TYPE=0` turns it off |
| compute queue families | Up to two compute and transfer queue families, for engines that want a family of their own for async compute | `KK_COMPUTE_QUEUES=n[,m]` |
| wsi/metal: rgba8 swapchains | `R8G8B8A8` swapchains, presented through a BGRA8 layer | |

## Performance

- Compiled shaders are kept across runs (Mesa's disk cache, under `~/.cache/mesa_shader_cache`): DOOM: The Dark
  Ages reaches its menu shell in about 2 s instead of 10 s from the second launch on.
- Small animated BLASes build fast without Metal's refit layout; ray queries that never inspect candidates let the
  hardware cull them; whole-subgroup reductions are native Metal `simd_*` instructions.
- Less CPU per frame: dynamic depth/stencil states are cached, the residency set is only committed when it changes,
  sparse binds are batched, waits on any of several semaphores sleep instead of polling.
- MetalFX objects are freed when FSR or DLSS is turned off and when the output resolution changes.

Everything above keeps the rendered output identical except Metal's own summation order for subgroup float sums;
`KK_EXACT=1` turns off any optimization that can change output.

## Building

On an Apple silicon Mac with macOS 26 or later (KosmicKrisp needs Metal 4):

1. Install the Xcode command line tools and the build dependencies with [Homebrew](https://brew.sh):

   ```sh
   xcode-select --install
   brew install llvm spirv-llvm-translator spirv-tools glslang bison ninja pkgconf python
   ```

2. Build:

   ```sh
   ./build-kosmickrisp.sh            # arm64, for native apps
   ./build-kosmickrisp.sh x86_64     # for apps under Rosetta, such as Wine/CrossOver games
   ./build-kosmickrisp.sh universal  # both in one library
   ```

   The script puts Meson and Mesa's Python modules in a virtual environment under `build-kk/`. For x86_64 it builds
   Mesa's compiler tools natively first, then cross-builds the driver. A first build takes a few minutes.

3. The output is in `dist/<target>/`:

   | File | What it is |
   |---|---|
   | `libvulkan_kosmickrisp.dylib` | The Vulkan driver |
   | `kosmickrisp_icd.json` | Its Vulkan ICD manifest, pointing at the dylib |

   Use it through the Vulkan loader, for example
   `VK_DRIVER_FILES=$PWD/dist/arm64/kosmickrisp_icd.json vkcube`. Wine/CrossOver games need the x86_64 or
   universal build and a Vulkan loader in the Wine environment; set `VK_DRIVER_FILES` there.

To build by hand instead, the native build is upstream KosmicKrisp's:

```sh
meson setup build -Dbuildtype=release -Dplatforms=macos -Dvulkan-drivers=kosmickrisp -Dgallium-drivers= \
  -Dopengl=false -Dzstd=disabled -Dmesa-clc=enabled -Dprecomp-compiler=enabled
ninja -C build src/kosmickrisp/vulkan/libvulkan_kosmickrisp.dylib
```

## Tested

On an M3 Max (macOS 27), x86_64 builds under CrossOver 26:

- **DOOM Eternal**: runs; DLSS Super Resolution through MetalFX upscaling, with a DLSS runtime that drives the channel.
- **DOOM: The Dark Ages**: runs with its mandatory ray tracing; FSR 3.1 upscaling and frame generation through MetalFX,
  with an FSR runtime that drives the channel (FSR frame generation needs `KK_QUEUE_COUNT=2 KK_COMPUTE_QUEUES=2,1`
  for the game to offer it). Around 16-20 fps at 3456x2234, about 30 with frame generation.

Settings used for both: `KK_RAY_TRACING=1 KK_SPARSE=1 KK_NVX=1 KK_VENDOR_ID=nvidia KK_DRIVER_ID=nvidia
KK_REPORT_EXTENSIONS=VK_NV_mesh_shader`.

## Limitations

- Ray tracing: no position fetch, any-hit doesn't run for procedural groups, serialized acceleration structures are
  rebuilt rather than restored. Performance is well below native; in DOOM: The Dark Ages the GPU is the limit.
- Sparse: buffers and 2D images only, no 3D or multisampled sparse images.
- MetalFX upscaling stops at 3x per axis (MetalFX's limit). Frame interpolation makes one frame between two real ones.
- MetalFX only does something when a runtime on the game's side uses the channel; the driver doesn't replace DLSS
  or FSR by itself. DLSS through NVIDIA Streamline (DOOM: The Dark Ages) doesn't reach it yet.
- Reporting another vendor or features (`KK_VENDOR_ID`, `KK_REPORT_*`) only changes what games are told.
- Based on the 26.3 branch; not yet rebased onto Mesa's main, where KosmicKrisp has moved on.

## License

MIT, like the rest of KosmicKrisp. Other parts of Mesa keep their own licenses, see [docs/license.rst](docs/license.rst).
