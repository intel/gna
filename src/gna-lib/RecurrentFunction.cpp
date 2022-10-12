/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "RecurrentFunction.h"

#include "AccelerationDetector.h"
#include "ActivationFunction.h"
#include "AffineLayerCapabilities.h"
#include "Bias.h"
#include "Capabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "LayerConfiguration.h"
#include "OperationConfig.h"
#include "Shape.h"
#include "Validator.h"
#include "Weight.h"

#include "gna2-common-api.h"

#include <algorithm>
#include <cstdint>
#include <memory>

using namespace GNA;

const FullCapabilitiesMap RecurrentFunction::outputCapabilities =
LayerCapabilities::MakeFullCaps<OutputOperandIndex>(INTEL_RECURRENT);

// Could not split into separate methods for each component as multibias weight scaling is using bias' and weights; tensors...
std::unique_ptr<RecurrentFunction> RecurrentFunction::Create(
    const TransformFactoryConfig& config,
    const OperationConfig& operationConfig)
{
    auto delay = operationConfig.FeedbackDelay;
    auto kernelOperation = operationConfig.GetKernelOperation();
    auto weightTensor = operationConfig.WeightsTensor;
    auto biasTensor = operationConfig.BiasesTensor;

    const std::function<void()> command = [&]()
    {
        const auto weights = Shape::Create(weightTensor.Shape, GNA_TENSOR_HW);
        const auto expectedWeights = Shape{ GNA_TENSOR_HW, config.output->Dimensions.at('W'), config.output->Dimensions.at('W') + config.input->Dimensions.at('W') };
        weights.ExpectEqual(expectedWeights);
    };
    ModelErrorHelper::ExecuteForModelItem(command, WeightOperandIndex);

    auto weights = std::make_unique<const WeightTensor>(weightTensor, config.validator);
    auto biases = std::make_unique<const BiasTensor>(
        biasTensor, 0, Gna2BiasModeDefault, config.validator);
    auto const kernelMode = KernelMode{ config.input->Mode, weights->Mode, biases->Mode };
    const auto& affineKernel = AccelerationDetector::GetKernelMap<RecurrentKernel>(
        static_cast<kernel_op>(kernelOperation), kernelMode);

    auto configCopy = config;
    configCopy.input = configCopy.output;
    auto activation = ActivationFunction::Create(configCopy);

    auto recurrentFunction = std::make_unique<RecurrentFunction>(
        BaseTransformConfig<RecurrentKernel>{config, affineKernel},
        operationConfig.GetTransformOperation(),
        delay, std::move(weights), std::move(biases), std::move(activation));

    return recurrentFunction;
}

void RecurrentFunction::ValidateFeedbackDelay() const
{
    auto const ctx = ModelItem{ Gna2ItemTypeParameter, Gna2DisabledU32, ModelWrapper::GetOperationInfo(Gna2OperationTypeRecurrent, ParameterIndexDelay) };
    ModelErrorHelper::ExpectAboveEq(FeedbackDelay, 1u, ctx);
    ModelErrorHelper::ExpectBelowEq(FeedbackDelay, Input->Dimensions.at('H'), ctx);
}

RecurrentFunction::RecurrentFunction(
    const BaseTransformConfig<RecurrentKernel>& config,
    TransformOperation transform, uint32_t delay,
    std::unique_ptr<const WeightTensor> weights,
    std::unique_ptr<const BiasTensor> biases,
    std::unique_ptr<ActivationFunction> activation) :
    Transform{ transform, &config.kernels, config.input },
    Weights{ std::move(weights) },
    Biases{ std::move(biases) },
    FeedbackDelay{ delay },
    Activation{ std::move(activation) }
{
    ValidateFeedbackDelay();

    Output = std::make_unique<OutputTensor>(
        Shape{ GNA_TENSOR_HW, config.output->Dimensions.at('H'), config.output->Dimensions.at('W') },
        config.output->Mode, config.outputBuffer,
        config.validator, outputCapabilities);
    Expect::Equal(Input->Dimensions.at('H'), Output->Dimensions.at('H'), Gna2StatusXnnErrorLyrCfg);

    auto const feedbackBuffer = CalculateFeedbackBuffer(config.output->Buffer);
    auto const kernelRecurrentConfig = RecurrentConfig{ config.output->Dimensions.at('W'),
        config.input->Dimensions.at('H'), config.input->Dimensions.at('W'),
        config.input->Buffer, feedbackBuffer, config.outputBuffer, config.output->Buffer,
        *Weights, *Biases, Biases->Mode.Size, config.output->Mode.Size,
        { Output->at(GNA_DIM_W), Activation->Pwl.get() } };

    hiddenConfig = std::make_unique<KernelConfig<RecurrentConfig>>(kernelRecurrentConfig,
        BaseConfig{ Input->Buffer, Output->Buffer });
}

void RecurrentFunction::UpdateConfigBuffers(
    std::unique_ptr<BaseConfig> configs[TransformOperationCount],
    const BufferMap& buffers) const
{
    Transform::UpdateConfigBuffers(configs, buffers);

    if (buffers.count(OutputOperandIndex) != 0)
    {
        auto config = GetConfig(configs);
        config->Transform.feedbackBuffer = CalculateFeedbackBuffer(buffers.at(OutputOperandIndex));
        config->Transform.output = hiddenConfig->Transform.output;
    }
}

const BaseAddress RecurrentFunction::CalculateFeedbackBuffer(const BaseAddress& outputBuffer) const
{
    if (outputBuffer)
    {
        auto delaySize = (FeedbackDelay * Output->Dimensions.at('W') * Output->Mode.Size);
        const auto buffer = outputBuffer - delaySize;

        try
        {
            Output->ValidateBuffer(buffer);
        }
        catch (const GnaException&)
        {
            throw GnaException(Gna2StatusXnnErrorNoFeedback);
        }
        return buffer;
    }

    return BaseAddress();
}

const ActivationFunction & RecurrentFunction::GetActivationFunction() const
{
    return *Activation;
}

Tensor const& RecurrentFunction::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case WeightOperandIndex:
    {
        return GetOperandIfExistOrThrow(Weights);
    }
    case BiasOperandIndex:
    {
        return GetOperandIfExistOrThrow(Biases);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}
