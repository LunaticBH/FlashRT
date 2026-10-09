// ================================================================
// Gapless FlashRT fork: Programmatic Dependent Launch (PDL) for the
// hand-written kernels on the Pi0.5 cuBLASLt path.
//
// Upstream FlashRT uses PDL only in the sm_120a skinny decoder
// (csrc/kernels/pi05/pi05_decoder_skinny_fp8_sm120.cu: pdl_prologue,
// launch_ex). This header brings the same two mechanisms to the generic
// kernels in norm.cu, fusion.cu, activation.cu, rope.cu, quantize.cu,
// elementwise.cu, dit_bf16.cu and patch_embed.cu that pipeline_rtx.py
// launches on every architecture:
//
//   * device side: fvk_pdl::trigger() lets the next kernel in the stream
//     launch early (griddepcontrol.launch_dependents) and fvk_pdl::wait()
//     blocks until every preceding kernel has completed and flushed its
//     memory (griddepcontrol.wait). Both compile to nothing below sm_90,
//     so one source serves sm_89 (no overlap, correctness only) and
//     sm_110 (overlap).
//   * host side: fvk_pdl::launch() replaces <<<>>> with cudaLaunchKernelEx
//     and attaches cudaLaunchAttributeProgrammaticStreamSerialization when
//     GAPLESS_FLASHRT_PDL=1 is set in the environment. Under stream capture
//     the attribute becomes a programmatic graph edge
//     (cudaGraphDependencyTypeProgrammatic), which the Gapless FlashRT
//     probe (scripts/pdl_probe.cu) confirmed on CC 8.9 with CUDA 13.0.
//
// Contract for every kernel launched through FVK_PDL_LAUNCH: before
// fvk_pdl::wait() it may read only data that no kernel in the graph
// writes (weights, biases, styles, RoPE tables, calibrated scales) and
// it must not write anything. Reads of activations and every store come
// after the wait. A kernel that does not follow this contract must keep
// its <<<>>> launch: with the attribute it would read stale data on
// sm_90+ while passing every test on sm_89.
//
// With the variable unset (the default) cudaLaunchKernelEx carries no
// attribute and the captured graph is the stock one, so the baseline
// configuration of the Gapless FlashRT harness is unchanged by this
// header.
// ================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <utility>

namespace fvk_pdl {

// ── Device side ──
__device__ __forceinline__ void trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}

// The usual prologue: let the dependent grid launch, then wait for the
// primary. Kernels that read weights before the wait call trigger() and
// wait() separately.
__device__ __forceinline__ void prologue() {
    trigger();
    wait();
}

// ── Host side ──
// Read once per process. "1" enables the attribute; anything else leaves
// the launches stock. harness.configs.apply_env sets the variable before
// the frontend is built, so the first captured graph already sees it.
inline bool enabled() {
    static const bool value = [] {
        const char* v = std::getenv("GAPLESS_FLASHRT_PDL");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return value;
}

template <typename... ExpTypes, typename... ActTypes>
inline cudaError_t launch(void (*kernel)(ExpTypes...), dim3 grid, dim3 block, size_t smem,
                          cudaStream_t stream, ActTypes&&... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = enabled() ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, kernel, std::forward<ActTypes>(args)...);
}

}  // namespace fvk_pdl

// Drop-in for `kernel<<<grid, block, smem, stream>>>(args...)`. The kernel
// must call fvk_pdl::wait() (or prologue()) as described above.
#define FVK_PDL_LAUNCH(kernel, grid, block, smem, stream, ...) \
    (void)::fvk_pdl::launch(kernel, dim3(grid), dim3(block), static_cast<size_t>(smem), stream, \
                            __VA_ARGS__)
