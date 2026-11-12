#!/bin/bash
# build-kosmickrisp.sh: builds the KosmicKrisp Vulkan driver from this tree.
#
#   ./build-kosmickrisp.sh [arm64|x86_64|universal] [output dir]
#
#   arm64      native Apple silicon apps (default)
#   x86_64     apps under Rosetta, such as Wine/CrossOver (cross-built: Mesa's compiler tools are built
#              natively first, then the driver for x86_64)
#   universal  both in one dylib
#
# Output (default ./dist/<target>): libvulkan_kosmickrisp.dylib and kosmickrisp_icd.json, a Vulkan ICD
# manifest pointing at it. Use it with VK_DRIVER_FILES=/path/to/kosmickrisp_icd.json.
#
# Needs the Xcode command line tools and Homebrew packages:
#   brew install llvm spirv-llvm-translator spirv-tools glslang bison ninja pkgconf python
# (glslangValidator can come from the Vulkan SDK instead of glslang). The driver needs macOS 26 (Metal 4).
# Python packages (meson, mako, pyyaml, packaging) go into a virtual environment in the build directory.
set -euo pipefail

target="${1:-arm64}"
src="$(cd "$(dirname "$0")" && pwd)"
out="$(mkdir -p "${2:-$src/dist/$target}" && cd "${2:-$src/dist/$target}" && pwd)"
build="$src/build-kk"

case "$target" in arm64|x86_64|universal) ;; *) echo "usage: $0 [arm64|x86_64|universal] [output dir]" >&2; exit 1 ;; esac
[ "$(uname -m)" = arm64 ] || { echo "error: build on an Apple silicon Mac" >&2; exit 1; }
xcrun --find clang >/dev/null 2>&1 || { echo "error: needs the Xcode command line tools: xcode-select --install" >&2; exit 1; }
brew="$(command -v brew || echo /opt/homebrew/bin/brew)"
[ -x "$brew" ] || { echo "error: needs Homebrew (https://brew.sh)" >&2; exit 1; }
missing=""
for formula in llvm spirv-llvm-translator spirv-tools bison ninja pkgconf; do
  "$brew" list --formula "$formula" >/dev/null 2>&1 || missing="$missing $formula"
done
command -v glslangValidator >/dev/null 2>&1 || "$brew" list --formula glslang >/dev/null 2>&1 || missing="$missing glslang"
[ -z "$missing" ] || { echo "error: missing Homebrew packages; run: brew install$missing" >&2; exit 1; }

hb="$("$brew" --prefix)"
export PATH="$hb/opt/llvm/bin:$hb/opt/bison/bin:$hb/bin:$PATH"
export PKG_CONFIG_PATH="$hb/lib/pkgconfig:$hb/opt/llvm/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export MACOSX_DEPLOYMENT_TARGET=26.0

mkdir -p "$build"
if [ ! -x "$build/venv/bin/meson" ]; then
  python3 -m venv "$build/venv"
  "$build/venv/bin/pip" install --quiet meson mako pyyaml packaging
fi
export PATH="$build/venv/bin:$PATH"

common=(-Dbuildtype=release -Db_ndebug=true -Dplatforms=macos -Dvulkan-drivers=kosmickrisp -Dgallium-drivers=
        -Dopengl=false -Dzstd=disabled -Dbuild-tests=false)

# Native: the arm64 driver, plus Mesa's compiler tools that the x86_64 build runs
buildNative() {
  if [ ! -f "$build/native/build.ninja" ]; then
    meson setup "$build/native" "$src" "${common[@]}" -Dprefix="$build/tools" \
      -Dmesa-clc=enabled -Dinstall-mesa-clc=true -Dprecomp-compiler=enabled -Dinstall-precomp-compiler=true
  fi
  ninja -C "$build/native"
  ninja -C "$build/native" install >/dev/null
}

buildX86() {
  buildNative
  cat > "$build/x86_64-darwin.ini" <<'EOF'
[binaries]
c = ['clang', '-arch', 'x86_64', '-mmacosx-version-min=26.0']
cpp = ['clang++', '-arch', 'x86_64', '-mmacosx-version-min=26.0']
objc = ['clang', '-arch', 'x86_64', '-mmacosx-version-min=26.0']
objcpp = ['clang++', '-arch', 'x86_64', '-mmacosx-version-min=26.0']
ar = 'ar'
strip = 'strip'
pkg-config = 'false'

[properties]
needs_exe_wrapper = true

[host_machine]
system = 'darwin'
subsystem = 'macos'
kernel = 'xnu'
cpu_family = 'x86_64'
cpu = 'x86_64'
endian = 'little'
EOF
  if [ ! -f "$build/x86_64/build.ninja" ]; then
    # Apple's clang for the target; the compiler tools from the native build
    PATH="$build/tools/bin:/usr/bin:$PATH" meson setup "$build/x86_64" "$src" "${common[@]}" \
      --cross-file "$build/x86_64-darwin.ini" -Dllvm=disabled -Dmesa-clc=system -Dprecomp-compiler=system \
      -Dxmlconfig=disabled -Dexpat=disabled
  fi
  PATH="$build/tools/bin:/usr/bin:$PATH" ninja -C "$build/x86_64" src/kosmickrisp/vulkan/libvulkan_kosmickrisp.dylib
}

dylib=src/kosmickrisp/vulkan/libvulkan_kosmickrisp.dylib
case "$target" in
  arm64) buildNative; cp "$build/native/$dylib" "$out/" ;;
  x86_64) buildX86; cp "$build/x86_64/$dylib" "$out/" ;;
  universal) buildX86; lipo -create "$build/native/$dylib" "$build/x86_64/$dylib" -output "$out/libvulkan_kosmickrisp.dylib" ;;
esac
codesign -f -s - "$out/libvulkan_kosmickrisp.dylib" 2>/dev/null || true

# The manifest Mesa generates, pointed at the copy in the output directory
manifest="$(ls "$build"/native/src/kosmickrisp/vulkan/kosmickrisp_mesa_icd.*.json | head -1)"
python3 - "$manifest" "$out" <<'EOF'
import json, sys
manifest = json.load(open(sys.argv[1]))
manifest["ICD"]["library_path"] = sys.argv[2] + "/libvulkan_kosmickrisp.dylib"
json.dump(manifest, open(sys.argv[2] + "/kosmickrisp_icd.json", "w"), indent=4)
EOF
echo
echo "built: $out/libvulkan_kosmickrisp.dylib ($(lipo -archs "$out/libvulkan_kosmickrisp.dylib"))"
echo "use:   VK_DRIVER_FILES=$out/kosmickrisp_icd.json"
