/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "SoftwareModel.h"

#include "Expect.h"
#include "HardwareCapabilities.h"
#include "KernelArguments.h"
#include "Layer.h"
#include "Macros.h"
#include "ModelError.h"
#include "RequestConfiguration.h"
#include "Validator.h"


#include <cstdint>
#include <functional>
#include <map>
#include <utility>

using namespace GNA;

void SoftwareModel::buildSingleLayer(std::unique_ptr<Layer> & layer)
{
    if (!layer)
    {
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }

    for (auto && operand : maximumOperandSizes)
    {
        FindMaximumOperandSizeForSingleLayer(*layer, operand.first, operand.second);
    }

    layer->VerifyHas1BInputAnd2BWeight();

    layers.push_back(std::move(layer));

    bufferConfigValidator.populate(*layers.back(), static_cast<uint32_t>(layers.size()) - 1);
}

uint32_t SoftwareModel::FindMaximumOperandSize(uint32_t operandIndex) const
{
    uint32_t maxSize = 0;
    for (auto const & layer : layers)
    {
        auto const operand = layer->TryGetOperand(operandIndex);
        if (nullptr != operand)
        {
            auto const bufferSize = operand->Size;
            maxSize = ((maxSize) > (bufferSize)) ? (maxSize) : (bufferSize);
        }
    }
    return maxSize;
}

void SoftwareModel::FindMaximumOperandSizeForSingleLayer(Layer const & layer,
    uint32_t operandIndex, uint32_t& maxSize)
{
    auto operandSize = layer.TryGetOperandSize(operandIndex);
    maxSize = ((maxSize) > (operandSize)) ? (maxSize) : (operandSize);
}

SoftwareModel::SoftwareModel(const Gna2Model& model, BaseValidator const&& softwareOnlyValidator,
    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerationsIn) :
    SoftwareModel{ model, supportedCpuAccelerationsIn }
{
    build(model.Operations, softwareOnlyValidator, softwareOnlyValidator, {});
}

SoftwareModel::SoftwareModel(const Gna2Model& model,
    BaseValidator const && softwareOnlyValidator,
    BaseValidator const && hwConsistentValidator,
    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerationsIn,
    const std::vector<std::unique_ptr<SubModel>>& subModels) :
    SoftwareModel{ model, supportedCpuAccelerationsIn }
{
    build(model.Operations, softwareOnlyValidator, hwConsistentValidator, subModels);
}

SoftwareModel::SoftwareModel(const Gna2Model& model,
    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerationsIn) :
    layerCount{ model.NumberOfOperations },
    supportedCpuAccelerations{ supportedCpuAccelerationsIn }
{
    Expect::NotNull(model.Operations);
}

void SoftwareModel::build(const Gna2Operation* const operations, const BaseValidator& softwareOnlyValidator,
    const BaseValidator& hwConsistentValidator, const std::vector<std::unique_ptr<SubModel>>& subModels)
{
    maximumOperandSizes.emplace(ScratchpadOperandIndex, 0);
    maximumOperandSizes.emplace(SoftwareScratchpadOperandIndex, 0);
    const auto hasHwValidator = !subModels.empty()
        && softwareOnlyValidator.Generation != hwConsistentValidator.Generation;

    for (auto i = uint32_t{ 0 }; i < layerCount; i++)
    {
        try
        {
            auto * validator = &softwareOnlyValidator;
            if (hasHwValidator && !SubModel::IsSoftwareLayer(i, subModels))
            {
                validator = &hwConsistentValidator;
            }
            auto layer = Layer::Create(operations[i], *validator);
            buildSingleLayer(layer);
        }
        catch (...)
        {
            GnaModelErrorException::DispatchAndSetLayer(i);
        }
    }
}

void SoftwareModel::Score(ScoreContext & context)
{
    const auto accel = context.requestConfiguration.Acceleration.GetEffectiveSoftwareAccelerationMode(supportedCpuAccelerations);

    LogAcceleration(accel);

    context.buffers->ReallocateCnnScratchPad(maximumOperandSizes.at(SoftwareScratchpadOperandIndex));
    auto config = InferenceConfig{ context.buffers, context.requestConfiguration };
    auto layerIter = layers.cbegin() + context.layerIndex;
    auto const layerEnd = layerIter + context.layerCount;

    context.profiler.Measure(Gna2InstrumentationPointLibExecution);

    for (; layerIter < layerEnd; ++layerIter)
    {
        auto const & layer = *layerIter;
        auto const found = context.requestConfiguration.LayerConfigurations.find(context.layerIndex);
        if (found == context.requestConfiguration.LayerConfigurations.end())
        {

            layer->ComputeHidden(accel, config.GetEffective(*layer));
        }
        else
        {
            auto const layerConfiguration = found->second.get();
            layer->Compute(*layerConfiguration, accel, config.GetEffective(*layer));
        }

        ++context.layerIndex;
    }

    context.saturationCount += config.SaturationCount;
}

uint32_t SoftwareModel::GetMaximumOperandSize(uint32_t operandIndex)
{
    auto const & found = maximumOperandSizes.find(operandIndex);
    if (maximumOperandSizes.cend() != found)
    {
        return found->second;
    }

    auto maxSize = FindMaximumOperandSize(operandIndex);
    maximumOperandSizes.emplace(operandIndex, maxSize);
    return maxSize;
}

Layer const& SoftwareModel::GetLayer(uint32_t layerIndex) const
{
    try
    {
        auto const & layer = layers.at(layerIndex);
        return *layer;
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

InferenceConfig::InferenceConfig(KernelBuffers* fvBuffers,
    RequestConfiguration const& requestConfiguration) :
    SaturationCount{ 0 }
{
    executionConfig = std::make_unique<ExecutionConfig>(fvBuffers,
        &SaturationCount, requestConfiguration.BufferElementCount);
    has3_0Consistency = HardwareCapabilities::Is3_0Device(requestConfiguration.GetConsistentDevice());
    if (has3_0Consistency)
    {
        executionConfig3_0 = std::make_unique<ExecutionConfig>(fvBuffers,
            &SaturationCount, requestConfiguration.BufferElementCountFor3_0);
        getEffective = &InferenceConfig::getFor3_0Fix;
    }
    else
    {
        executionConfig3_0 = nullptr;
        getEffective = &InferenceConfig::getNormal;
    }
}

ExecutionConfig& InferenceConfig::getFor3_0Fix(Layer const & layer) const
{
    if (layer.Is1BInputAnd2BWeight())
    {
        return *executionConfig3_0;
    }
    return *executionConfig;
}

ExecutionConfig& InferenceConfig::getNormal(Layer const & layer) const
{
    UNREFERENCED_PARAMETER(layer);
    return *executionConfig;
}
