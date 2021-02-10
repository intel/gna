/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "LayerCapabilities.h"

namespace GNA
{

struct ConvolutionalLayer2DCapabilities : LayerCapabilities
{
    static const FullCapabilitiesMap & GetOperands(uint32_t operandIndex);
    static const FullCapabilitiesMap & GetParameters(uint32_t parameterIndex);

    static const OperationCapabilityMap & GetOperands(uint32_t operandIndex, nn_operation operation);
    static const OperationCapabilityMap & GetParameters(uint32_t parameterIndex, nn_operation operation);

    /** CNN minimum number of filter coefficients */
    static constexpr uint32_t Filter1DElementsMin = 8;

    /** CNN maximum number of filter coefficients */
    static constexpr uint32_t Filter1DElementsMax = 768;

    /** CNN 2D minimum number of kernel elements in one dimension */
    static constexpr uint32_t Filter2DElementsMin = 1;

    /** CNN 2D maximum number of kernel elements in one dimension */
    static constexpr uint32_t Filter2DElementsMax = 255;

    /** CNN number of filter coefficients constraint - must be multiple of */
    static constexpr uint32_t Filter1DElementsMultiplier = 4;

    /** CNN maximum number of filters */
    static constexpr uint32_t Filter1DCountMax = ((UINT16_MAX + 1) - 4);

    /** CNN 2D maximum number of kernels */
    static constexpr uint32_t Filter2DCountMax = 8192;

    /** CNN minimum size of pooling window */
    static constexpr uint32_t PoolingWindowSizeMin = 1;

    /** CNN maximum size of pooling window */
    static constexpr uint32_t PoolingWindowSizeMax = 6;
};

}
