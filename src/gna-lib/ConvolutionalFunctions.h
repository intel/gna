/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Address.h"
#include "Bias.h"
#include "Component.h"
#include "DataMode.h"
#include "KernelArguments.h"
#include "Shape.h"
#include "Weight.h"
#include "XnnKernel.h"

#include <map>
#include <memory>

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;
struct Tensor;

struct FiltersTensor : public WeightTensor
{
    static std::unique_ptr<const FiltersTensor> Create(const Gna2Tensor& filtersTensor, const LayerValidator& validator);
    FiltersTensor(const Shape& dimensions, const DataMode& dataMode, void * buffer, const LayerValidator& validator);
    virtual ~FiltersTensor() = default;

    uint32_t Count;
    uint32_t CoefficientCount;
};

struct ConvolutionFunction
{

    static std::unique_ptr<const ConvolutionFunction> Create(const Tensor * input, const Tensor * output,
        const Gna2Operation& operationDetails, const LayerValidator & validatorIn)
    {
        expectValid(operationDetails);
        auto filters = createFilters(operationDetails, validatorIn);
        auto stride = createStride(operationDetails, validatorIn);
        auto biases = createBiases(operationDetails, validatorIn);

        return finalizeCreation(input, output,
            std::move(filters), std::move(stride), std::move(biases));
    }

    ConvolutionFunction(const KernelMap<ConvolutionKernel>& kernelsIn,
        const Tensor* input, const Tensor* output, std::unique_ptr<const FiltersTensor> filters,
        std::unique_ptr<const BiasTensor> biases, std::unique_ptr<const Component> stride);

    std::unique_ptr<const ConvolutionConfig> GetRequestConfig(const BaseAddress& inputs, const BaseAddress& outputs) const;
    const ConvolutionConfig * GetHiddenConfig() const
    {
        return hiddenConfig.get();
    }

    void ComputeHidden(AccelerationMode accel, ExecutionConfig const & execution) const;
    void Compute(const ConvolutionConfig* config, AccelerationMode accel, ExecutionConfig const & execution) const;

    std::unique_ptr<const BiasTensor> Biases;

    std::unique_ptr<const FiltersTensor> Filters;

    // Sizes of convolution filter stride in each dimension (in # of elements).
    std::unique_ptr<const Component> Stride;

    // Dimensions of outputs per filter after convolution.
    Shape Output;

    // Total number of elements in output tensor per filter after convolution.
    uint32_t OutputsPerFilterCount;

protected:
    const KernelMap<ConvolutionKernel>& kernels;

    std::unique_ptr<ConvolutionConfig> hiddenConfig;

    static const FullCapabilitiesMap strideLimits;

private:
    static std::unique_ptr<const ConvolutionFunction> finalizeCreation(const Tensor* input,
        const Tensor* output, std::unique_ptr<const FiltersTensor> filters,
        std::unique_ptr<const Component> stride, std::unique_ptr<const BiasTensor> biases);

    static std::unique_ptr<const FiltersTensor> createFilters(const Gna2Operation & apiOperation,
        const LayerValidator & validatorIn);

    static std::unique_ptr<const Component> createStride(const Gna2Operation & cnn,
        const LayerValidator & validatorIn);

    static std::unique_ptr<const BiasTensor> createBiases(const Gna2Operation & apiOperation,
        const LayerValidator & validatorIn);

    static void expectValid(const Gna2Operation & apiOperation);
};

}
