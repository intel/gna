/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "KernelArguments.h"
#include "ParameterLimits.h"
#include "PoolingMode.h"
#include "Shape.h"
#include "XnnKernel.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cstdint>
#include <map>
#include <memory>

namespace GNA
{
class LayerValidator;
struct PwlCached;

struct PoolingFunction
{
    static std::unique_ptr<const PoolingFunction> Create(void const * layerDetails,
        const Shape& inputDimensions, const LayerValidator& validatorIn, gna_data_mode inputMode);

    static std::unique_ptr<const PoolingFunction> Create(Gna2Operation const & apiOperation,
        const Shape & inputDimensions, const LayerValidator & validatorIn, gna_data_mode inputMode);

    PoolingFunction(nn_operation const operation, const Shape& inputDimensions,
        const Shape& window, const Shape& stride, PoolingMode mode,
        const KernelMap<ConvolutionPoolingKernel>& kernelsIn);
    ~PoolingFunction() = default;

    void Compute(const ConvolutionConfig * convolutionConfig, AccelerationMode accel,
                int64_t * poolScratchPad, const PwlCached * pwl) const;

    const KernelPoolingMode Mode;
    // Pooling window dimensions (in # of elements).
    const Shape Window;

    // Sizes of Pooling window stride in each dimension (in # of elements).
    const Shape Stride;

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
