/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Bias.h"
#include "Capabilities.h"
#include "Component.h"
#include "ConvolutionalFunctions.h"
#include "OperationConfig.h"
#include "Transform.h"
#include "XnnKernel.h"

#include <memory>

namespace GNA
{
class LayerValidator;

struct ConvolutionFunction2D : public Transform<ConvolutionConfig2D, ConvolutionKernel2D>
{
    static std::unique_ptr<ConvolutionFunction2D> Create(
        const TransformFactoryConfig & config,
        const OperationConfig& operationConfig);

    ConvolutionFunction2D(const BaseTransformConfig<ConvolutionKernel2D>& config,
        std::unique_ptr<const FiltersTensor> filters,
        std::unique_ptr<const BiasTensor> biases,
        std::unique_ptr<const Component> stride,
        std::unique_ptr<const Component> padding);

    virtual ~ConvolutionFunction2D() = default;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    virtual bool Is1D() const override
    {
        return is1D;
    }

    std::unique_ptr<const BiasTensor> Biases;

    std::unique_ptr<const FiltersTensor> Filters;

    std::unique_ptr<const Component> Stride;

    std::unique_ptr<const Component> Padding;

protected:
    static std::unique_ptr<ConvolutionFunction2D> create(
        const TransformFactoryConfig & config,
        const OperationConfig& operationConfig);

    static Shape CalculateBiasShape(Gna2BiasMode mode, uint32_t filterCount, Shape const & outputShape);

    static std::unique_ptr<const BiasTensor> CreateBiasTensor(
        Gna2Tensor const & apiTensor, Gna2BiasMode biasMode, uint32_t filtersCount,
        Shape const & outputShape, const LayerValidator & validatorIn);

    static Shape GetOutputShape(Shape const & inputShape,
        Shape const & filerShape, Shape const & strideShape, Shape const & paddingShape);

    virtual void updateExecutionKernelConfig(ExecutionKernelConfig<ConvolutionConfig2D> & config)
        const override
    {
        setSoftwareScratchPad(config);
    }

    bool is1D = false;

};
}
