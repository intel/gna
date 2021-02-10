/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once
#include <cstdint>

enum KernelPoolingMode
{
    KernelPoolingModeNone = 0,
    KernelPoolingModeMax = 1,
    KernelPoolingModeSum = 2,
};

struct PoolingConfig2D
{
    PoolingConfig2D(const uint32_t InputWidthIn, const uint32_t InputHeightIn,
        const uint32_t InputDepthIn, KernelPoolingMode ModeIn,
        const uint32_t StrideWidthIn, const uint32_t StrideHeightIn,
        const uint32_t WindowWidthIn, const uint32_t WindowHeightIn) :
        InputWidth{ InputWidthIn },
        InputHeight{ InputHeightIn },
        InputDepth{ InputDepthIn },
        Mode{ ModeIn },
        StrideWidth{ StrideWidthIn },
        StrideHeight{ StrideHeightIn },
        WindowWidth{ WindowWidthIn },
        WindowHeight{ WindowHeightIn }
    {
    }

    const uint32_t InputWidth;
    const uint32_t InputHeight;
    const uint32_t InputDepth;

    KernelPoolingMode Mode;
    const uint32_t StrideWidth;
    const uint32_t StrideHeight;

    const uint32_t WindowWidth;
    const uint32_t WindowHeight;
};

struct PoolingConfig
{
    PoolingConfig(PoolingConfig const * const source, int64_t * const bufferIn);
    PoolingConfig(KernelPoolingMode const mode, uint32_t const sizeIn, uint32_t const stepIn);

    const KernelPoolingMode Mode;
    uint32_t const Size;
    uint32_t const Step;
    int64_t * Buffer;
};