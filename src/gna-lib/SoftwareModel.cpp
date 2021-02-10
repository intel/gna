/**
 @copyright (C) 2019-2021 Intel Corporation
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

#include "gna-api-types-xnn.h"

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
}

void SoftwareModel::CheckModel(uint32_t declaredBatchSize, void * operationPointer) const
{
    Expect::InRange(declaredBatchSize, ui32_1, XNN_N_GROUP_MAX,
        Gna2StatusXnnErrorLyrCfg);

    ModelErrorHelper::ExpectGtZero(layerCount, Gna2ItemTypeModelNumberOfOperations);
    Expect::InRange(layerCount, ui32_1,
        HardwareCapabilities::GetMaximumLayerCount(DefaultDeviceVersion),
        Gna2StatusXnnErrorNetLyrNo);
    Expect::NotNull(operationPointer);
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

SoftwareModel::SoftwareModel(const Gna2Model& model,
    BaseValidator validator,
    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerationsIn) :
    layerCount{ model.NumberOfOperations },
    supportedCpuAccelerations{ supportedCpuAccelerationsIn }
{
    CheckModel(1, model.Operations);
    build(model.Operations, validator);
}

uint32_t SoftwareModel::Score(
    uint32_t layerIndex,
    uint32_t layerCountIn,
    RequestConfiguration const &requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *fvBuffers)
{
    validateConfiguration(requestConfiguration);

    const auto accel = requestConfiguration.Acceleration.GetEffectiveSoftwareAccelerationMode(supportedCpuAccelerations);

    LogAcceleration(accel);

    fvBuffers->ReallocateCnnScratchPad(maximumOperandSizes.at(SoftwareScratchpadOperandIndex));
    auto config = InferenceConfig{ fvBuffers, requestConfiguration };
    auto layerIter = layers.cbegin() + layerIndex;
    auto const layerEnd = layerIter + layerCountIn;

    profiler->Measure(Gna2InstrumentationPointLibExecution);

    for (; layerIter < layerEnd; ++layerIter)
    {
        auto const & layer = *layerIter;
        auto const found = requestConfiguration.LayerConfigurations.find(layerIndex);
        if (found == requestConfiguration.LayerConfigurations.end())
        {
            layer->ComputeHidden(accel, config.GetEffective(*layer));
        }
        else
        {
            auto const layerConfiguration = found->second.get();
            layer->Compute(*layerConfiguration, accel, config.GetEffective(*layer));
        }

        ++layerIndex;
    }

    return config.SaturationCount;
}

void SoftwareModel::validateConfiguration(const RequestConfiguration& configuration) const
{
    UNREFERENCED_PARAMETER(configuration);
}

uint32_t SoftwareModel::GetMaximumOperandSize(uint32_t operandIndex)
{
    auto const & found = maximumOperandSizes.find(operandIndex);
    if (maximumOperandSizes.cend() != found)
    {
        return found->second;
    }

    uint32_t maxSize = FindMaximumOperandSize(operandIndex);
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
    auto const is3_0 = HardwareCapabilities::Is3_0Device(requestConfiguration.GetConsistentDevice());
    has3_0Consistency = is3_0 && requestConfiguration.Acceleration.GetHwConsistency();
    if (has3_0Consistency)
    {
        executionConfig3_0 = std::make_unique<ExecutionConfig>(fvBuffers,
            &SaturationCount, requestConfiguration.BufferElementCountFor3_0);
        getEffective = &InferenceConfig::getFor3_0Fix;
    }
    else
    {
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
