/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once
#include <cstdint>

class KernelDataMode {
    uint32_t bytesPerElement;
public:
    explicit KernelDataMode(uint32_t bytesPerElementIn):bytesPerElement{ bytesPerElementIn }
    {
    }
    operator uint32_t() const
    {
        return bytesPerElement;
    }
};

enum KernelBiasMode
{
    KernelBiasModePerFilter,
    KernelBiasModePerStride,
    KernelBiasModeDisabled
};

struct ConvolutionConfig2D
{
    ConvolutionConfig2D(const uint32_t InputWidthIn, const uint32_t InputHeightIn,
        const uint32_t InputDepthIn, const uint32_t NumberOfFiltersIn,
        const uint32_t FilterWidthIn, const uint32_t FilterHeightIn,
        const uint32_t FilterDepthIn, const KernelDataMode FilterDataModeIn,
        const void* const FilterDataIn, const uint32_t StrideWidthIn,
        const uint32_t StrideHeightIn, const uint32_t ZeroPaddingWidthIn,
        const uint32_t ZeroPaddingHeightIn, const KernelBiasMode BiasModeIn,
        const KernelDataMode BiasDataModeIn, const void* const BiasDataIn) :
    InputWidth{ InputWidthIn },
        InputHeight{ InputHeightIn },
        InputDepth{ InputDepthIn },
        NumberOfFilters{ NumberOfFiltersIn },
        FilterWidth{ FilterWidthIn },
        FilterHeight{ FilterHeightIn },
        FilterDepth{ FilterDepthIn },
        FilterDataMode{ FilterDataModeIn },
        FilterData{ FilterDataIn },
        StrideWidth{ StrideWidthIn },
        StrideHeight{ StrideHeightIn },
        ZeroPaddingWidth{ ZeroPaddingWidthIn },
        ZeroPaddingHeight{ ZeroPaddingHeightIn },
        BiasMode{ BiasModeIn },
        BiasDataMode{ BiasDataModeIn },
        BiasData{ BiasDataIn }
    {
    }
    const uint32_t InputWidth;
    const uint32_t InputHeight;
    const uint32_t InputDepth;

    const uint32_t NumberOfFilters;
    const uint32_t FilterWidth;
    const uint32_t FilterHeight;
    const uint32_t FilterDepth;

    const KernelDataMode FilterDataMode;
    const void* const FilterData;

    const uint32_t StrideWidth;
    const uint32_t StrideHeight;

    const uint32_t ZeroPaddingWidth;
    const uint32_t ZeroPaddingHeight;

    const KernelBiasMode BiasMode;
    const KernelDataMode BiasDataMode;
    const void* const BiasData;
};
