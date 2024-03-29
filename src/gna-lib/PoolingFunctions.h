/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Component.h"
#include "KernelArguments.h"
#include "ParameterLimits.h"
#include "Shape.h"
#include "XnnKernel.h"

#include <cstdint>
#include <map>
#include <memory>

namespace GNA
{
    struct DataMode;
    class LayerValidator;
    struct PwlCached;

struct PoolingFunction
{
    static std::unique_ptr<const PoolingFunction> Create(Gna2Operation const & apiOperation,
        const Shape & inputDimensions, const LayerValidator & validatorIn, const DataMode & inputMode);

    PoolingFunction(nn_operation const operation, const Shape& inputDimensions,
        const Shape& window, const Shape& stride, KernelPoolingMode mode,
        const KernelMap<ConvolutionPoolingKernel>& kernelsIn);
    ~PoolingFunction() = default;

    void Compute(const ConvolutionConfig * convolutionConfig, AccelerationMode accel,
                int64_t * poolScratchPad, const PwlCached * pwl) const;

    const KernelPoolingMode Mode;


    // Pooling window dimensions (in # of elements).
    const Component Window;

    // Sizes of Pooling window stride in each dimension (in # of elements).
    const Component Stride;

    // Dimensions of output tensor after pooling (in # elements).
    Shape OutputDimensions;

    // Total number of elements in output tensor per filter after pooling.
    uint32_t OutputsPerFilterCount;
protected:
    const KernelMap<ConvolutionPoolingKernel>& kernels;
    const  std::unique_ptr<PoolingConfig> hiddenConfig;
    static const std::map<const nn_operation, const ShapeLimits> windowLimits;
    static const std::map<const nn_operation, const ShapeLimits> strideLimits;

    static void ExpectValid(Gna2Operation const & apiOperation);
};

}
