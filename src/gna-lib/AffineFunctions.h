/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "AccelerationDetector.h"
#include "Address.h"
#include "Bias.h"
#include "KernelArguments.h"
#include "Tensor.h"
#include "Transform.h"
#include "Weight.h"
#include "XnnKernel.h"

#include "gna2-inference-impl.h"

#include <cstdint>
#include <map>
#include <memory>

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;
class OperationConfig;

struct LayerConfiguration;

// AffineFunction interface
struct AffineFunction : public Transform<AffineConfig, AffineKernel>
{
public:
    virtual ~AffineFunction() = default;

    static std::unique_ptr<AffineFunction> Create(
        const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);

    Tensor const& GetOperand(uint32_t operandIndex) const override;

    std::unique_ptr<const WeightTensor> Weights;
    std::unique_ptr<const BiasTensor> Biases;

protected:
    AffineFunction(const BaseTransformConfig<AffineKernel>& config,
        TransformOperation operation,
        std::unique_ptr<const WeightTensor> weights,
        std::unique_ptr<const BiasTensor> biases);

    static const ShapeLimits outputDimensionsLimits;

    static const DataModeLimits outputModeLimits_0_9;

    static const TensorLimits outputLimits_0_9;

    static const DataModeLimits outputModeLimits_3;

    static const TensorLimits outputLimits_3;

private:
    static const std::map<Gna2OperationType, kernel_op> kernelOperationMap;
    static std::unique_ptr<AffineFunction> createAffineSingleFunction(
        const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);
    static std::unique_ptr<AffineFunction> createAffineMultiFunction(
        const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);
};

class AffineFunctionSingle : public AffineFunction
{
public:
    AffineFunctionSingle(BaseTransformConfig<AffineKernel> config,
        TransformOperation transform,
        std::unique_ptr<const WeightTensor> weights,
        std::unique_ptr<const BiasTensor> biases);

    virtual ~AffineFunctionSingle() = default;

    void ValidateActiveList(ActiveList const & activeList) const override;

    void Compute(AccelerationMode accel, LayerConfiguration const* layerConfiguration,
                 ExecutionConfig const& execution) const override;

private:
    static const FullCapabilitiesMap outputCapabilities;

    const KernelMap<AffineActiveListKernel>& kernelsAl;
};

class AffineFunctionMulti : public AffineFunction
{
public:
    AffineFunctionMulti(BaseTransformConfig<AffineKernel> config,
        TransformOperation transform,
        std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases,
        std::unique_ptr<const Tensor> weightScaleFactors);
    virtual ~AffineFunctionMulti() = default;

    Tensor const& GetOperand(uint32_t operandIndex) const override;

    const std::unique_ptr<const Tensor> WeightScaleFactors; // AffineFunctionMulti1B

    static const FullCapabilitiesMap Capabilities;

private:
    static const FullCapabilitiesMap outputCapabilities;
};
}
