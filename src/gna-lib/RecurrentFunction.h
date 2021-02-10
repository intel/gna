/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "AccelerationDetector.h"
#include "ActivationFunction.h"
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

class RecurrentFunction : public Transform<RecurrentConfig, RecurrentKernel>
{
public:
    static std::unique_ptr<RecurrentFunction> Create(
        const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);

    RecurrentFunction(const BaseTransformConfig<RecurrentKernel>& config,
        std::unique_ptr<const PwlCached> pwl,
        TransformOperation operation, uint32_t delay,
        std::unique_ptr<const WeightTensor> weights,
        std::unique_ptr<const BiasTensor> biases);

    virtual ~RecurrentFunction() = default;

    const BaseAddress CalculateFeedbackBuffer(const BaseAddress& outputBuffer) const;

    const ActivationFunction& GetActivationFunction() const;
    void SetActivationFunction(std::unique_ptr<ActivationFunction> activation);

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    std::unique_ptr<const WeightTensor> Weights;
    std::unique_ptr<const BiasTensor> Biases;

protected:
    static const ShapeLimits outputDimensionsLimits;

    static const DataModeLimits outputModeLimits_0_9;

    static const TensorLimits outputLimits_0_9;

    static const DataModeLimits outputModeLimits_3;

    static const TensorLimits outputLimits_3;

private:
    static const FullCapabilitiesMap outputCapabilities;
    static const std::map<Gna2OperationType, kernel_op> kernelOperationMap;

    virtual void UpdateConfigBuffers(
            std::unique_ptr<BaseConfig> configs[TransformOperationCount],
            const BufferMap& buffers) const override;

    static void ValidateActivation(const Gna2Tensor& activationTensor);
    void ValidateFeedbackDelay() const;

    const uint32_t FeedbackDelay;

    const std::unique_ptr<const PwlCached> pwl;
    const ActivationConfig activationConfig;
    std::unique_ptr<ActivationFunction> Activation;
};

}

