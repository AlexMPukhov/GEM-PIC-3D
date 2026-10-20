#!/bin/sh
# ============================================================================
#  install_cuda_pip.sh -- install NVIDIA's CUDA compiler (nvcc), runtime and cuFFT
#  WITHOUT root rights, from NVIDIA's packages on PyPI, into one directory.
#
#  usage:  sh tools/install_cuda_pip.sh [prefix]        (default: $HOME/cuda-pip)
#
#  Afterwards build GEM-PIC-3D for the GPU with the command printed at the end.
#  Needs: python3 with pip, and an NVIDIA driver >= 580 (CUDA 13; check the
#  "CUDA Version" shown by nvidia-smi: it must be 13.0 or higher). With an
#  older driver use a CUDA 12 toolkit instead (see docs/gpu.md).
#  Nothing here is specific to GEM-PIC-3D: no Kokkos, no wrapper compiler.
# ============================================================================
set -e
PREFIX=${1:-$HOME/cuda-pip}
VER=${CUDA_PIP_VERSION:-13.4}
echo "installing CUDA $VER (nvcc, runtime, headers) into $PREFIX"
python3 -m pip install --quiet --target "$PREFIX/site" \
  "nvidia-cuda-nvcc==$VER.*" "nvidia-cuda-runtime==$VER.*" "nvidia-cuda-crt==$VER.*" "nvidia-nvvm==$VER.*" \
  "nvidia-cuda-cccl"
python3 -m pip install --quiet --target "$PREFIX/cufft" "nvidia-cufft==12.*"
ROOT="$PREFIX/site/nvidia/cu13"
FFT="$PREFIX/cufft/nvidia/cu13"
for f in "$FFT"/include/cufft*.h; do ln -sf "$f" "$ROOT/include/"; done
ln -sf "$FFT/lib/libcufft.so.12" "$ROOT/lib/libcufft.so.12"
ln -sf libcufft.so.12 "$ROOT/lib/libcufft.so"
if [ ! -x "$ROOT/bin/nvcc" ]; then echo "nvcc not found in $ROOT/bin"; exit 1; fi
# the usual toolkit layout expected by CMake and by the linker
[ -e "$ROOT/lib64" ] || ln -s lib "$ROOT/lib64"
[ -e "$ROOT/lib/libcudart.so" ] || ln -s libcudart.so.13 "$ROOT/lib/libcudart.so"
ln -sfn "$ROOT" "$PREFIX/cuda"
"$PREFIX/cuda/bin/nvcc" --version | tail -2
cat <<EOF

CUDA toolkit ready in $PREFIX/cuda. Build GEM-PIC-3D for the GPU with
  make CUDA=1 NVCC=$PREFIX/cuda/bin/nvcc CUDA_ARCH="-arch=native" BUILD=build-gpu
and test on the GPU with
  build-gpu/tests
EOF
