/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "CompiledModel.h"

#include "DeviceManager.h"
#include "Expect.h"
#include "GnaException.h"
#include "HardwareCapabilities.h"
#include "Layer.h"
#include "Logger.h"
#include "Macros.h"
#include "Memory.h"
#include "Request.h"
#include "RequestConfiguration.h"
#include "SubModel.h"

#include "gna-api-types-xnn.h"
#include "profiler.h"

#include <algorithm>
#include <cstring>

using namespace GNA;

void CompiledModel::CopyData(void *address, size_t size) const
{
    allocations.CopyData(address, size);
}

void CompiledModel::BuildHardwareModel(DriverInterface &ddi)
{
    const auto& deviceSubmodels = getSubmodels(hwCapabilities);

    auto hasHardwareCompliantLayer =
        !(1 == deviceSubmodels.size() && SubmodelType::Software == deviceSubmodels.at(0)->Type);
    if (!hasHardwareCompliantLayer)
    {
        Log->Warning("None of model layers is compliant with selected hardware GNA device, "
            "only software processing is available for this model.\n");
    }
    else if (hwCapabilities.IsHardwareSupported())
    {
        hardwareModel = std::make_unique<HardwareModelScorable>(*this, ddi, hwCapabilities);
    }

    if (hardwareModel)
    {
        hardwareModel->Build(deviceSubmodels);
    }
}

uint32_t CompiledModel::GetMaximumOperandSize(uint32_t operandIndex)
{
    return softwareModel.GetMaximumOperandSize(operandIndex);
}

void CompiledModel::InvalidateHardwareRequestConfig(uint32_t configId) const
{
    if (hardwareModel)
    {
        hardwareModel->InvalidateConfig(configId);
    }
}

bool CompiledModel::IsHardwareEnforcedModeValid()
{
    if (!hardwareModel)
    {
        return false;
    }

    return IsFullyHardwareCompatible(hwCapabilities);
}

bool CompiledModel::IsFullyHardwareCompatible(const HardwareCapabilities& targetDevice)
{
    const auto& deviceSubmodels = getSubmodels(targetDevice);
    for (const auto& submodel : deviceSubmodels)
    {
        if (submodel->Type == Software)
        {
            return false;
        }
    }
    return true;
}

CompiledModel::AccelerationType CompiledModel::getEffectiveAccelerationMode(RequestConfiguration& config)
{
    const auto isSoftwareEffective = config.Acceleration.IsSoftwareEnforced() ||
        !hwCapabilities.IsHardwareSupported() ||
        (config.GetConsistentDevice() != hwCapabilities.GetDeviceVersion());
    if (isSoftwareEffective)
    {
        return EnforcedSoftware;
    }
    return Auto;
}

Gna2Status CompiledModel::Score(
    RequestConfiguration& config,
    RequestProfiler *profiler,
    KernelBuffers *buffers)
{
    auto saturationCount = uint32_t{ 0 };
    try
    {
        profiler->Measure(Gna2InstrumentationPointLibProcessing);
        auto const effectiveAcceleration = getEffectiveAccelerationMode(config);
        switch (effectiveAcceleration)
        {
        case EnforcedSoftware:
            saturationCount = softwareModel.Score(0, LayerCount, config, profiler, buffers);
            break;
        case Auto:
            saturationCount = scoreAllSubModels(config, profiler, buffers);
            break;
        default:
            return Gna2StatusAccelerationModeNotSupported;
            break;
        }
        profiler->Measure(Gna2InstrumentationPointLibCompletion);
    }
    catch (const GnaException& e)
    {
        return e.GetStatus();
    }
    catch (...)
    {
        Log->Error("Unknown Exception in CompiledModel::Score()\n");
        return Gna2StatusUnknownError;
    }
    return (saturationCount > 0) ? Gna2StatusWarningArithmeticSaturation : Gna2StatusSuccess;
}

void CompiledModel::ValidateBuffer(MemoryContainer const & requestAllocations, Memory const & memory) const
{
    if (hardwareModel)
    {
        hardwareModel->ValidateConfigBuffer(requestAllocations, memory);
    }
}
Memory const & CompiledModel::getMemoryFromDeviceAllocations(const void *buffer, size_t bufferSize) const
{
    const auto& allAllocations = DeviceManager::Get().GetAllAllocated();

    for (auto const & memory : allAllocations)
    {
        Expect::NotNull(memory.get(), Gna2StatusXnnErrorInvalidBuffer);

        auto const memoryBuffer = memory->GetBuffer();
        Expect::NotNull(memoryBuffer, Gna2StatusXnnErrorInvalidBuffer);

        auto const memorySize = memory->GetSize();
        if (Expect::InMemoryRange(buffer, bufferSize, memoryBuffer, memorySize))
        {
            return *memory;
        }
    }
    throw GnaException(Gna2StatusXnnErrorInvalidBuffer);
}

Memory const * CompiledModel::GetMemoryIfNotPartOfModel(const void *buffer, size_t bufferSize) const
{
    if (allocations.Contains(buffer, bufferSize)) // already part of a model allocations, no further action is needed
    {
        return nullptr;
    }

    return &getMemoryFromDeviceAllocations(buffer, bufferSize);
}

void CompiledModel::VerifyBufferAndStoreMemory(const void *buffer, size_t bufferSize)
{
    if (!allocations.Contains(buffer, bufferSize))
    {
        auto const & memory = getMemoryFromDeviceAllocations(buffer, bufferSize);
        allocations.Emplace(memory);
    }
}

BaseValidator CompiledModel::makeValidator()
{
    return BaseValidator
    {
        HardwareCapabilities(),
        ValidBoundariesFunctor {
            [this](const void *buffer, size_t bufferSize)
            {
                try
                {
                    VerifyBufferAndStoreMemory(buffer, bufferSize);
                }
                catch (GnaException&)
                {
                    throw GnaModelErrorException(Gna2ItemTypeOperandData,
                        Gna2ErrorTypeArgumentInvalid, reinterpret_cast<int64_t>(buffer));
                }
            }
        }
    };
}

uint32_t CompiledModel::scoreAllSubModels(RequestConfiguration& config,
    RequestProfiler *profiler, KernelBuffers *buffers)
{
    const auto& deviceSubmodels = getSubmodels(hwCapabilities);
    auto saturationCount = uint32_t{ 0 };
    for (const auto& submodel : deviceSubmodels)
    {
        uint32_t layerIndex = submodel->LayerIndex;
        uint32_t layerCount = submodel->GetLayerCount();
        switch (submodel->Type)
        {
        case Software:
            saturationCount += softwareModel.Score(layerIndex, layerCount, config, profiler, buffers);
            break;
        case Hardware:
        case GMMHardware:
            saturationCount += hardwareModel->Score(layerIndex, layerCount, config, profiler, buffers);
            break;
        }
    }
    return saturationCount;
}

const std::vector<std::unique_ptr<SubModel>>&
CompiledModel::getSubmodels(const HardwareCapabilities& hwCaps)
{
    if (submodels.find(hwCaps.GetDeviceVersion()) == submodels.end())
    {
        createSubmodels(hwCaps);
    }

    return submodels.at(hwCaps.GetDeviceVersion());
}

SubmodelType CompiledModel::getSubmodelType(
    const HardwareCapabilities &hwCaps, uint32_t layerIndex) const
{
    auto const & layer = softwareModel.GetLayer(layerIndex);
    auto deviceGeneration = hwCaps.GetDeviceGeneration();
    auto dataConfig = layer.GetDataMode();
    auto supportMapIterator = DataConfig::Capabilities.find(dataConfig);
    if (supportMapIterator == DataConfig::Capabilities.end())
    {
        return SubmodelType::Software;
    }

    const auto& supportMap = supportMapIterator->second;
    auto supportIterator = supportMap.find(layer.Operation);
    if (supportIterator == supportMap.end())
    {
        return SubmodelType::Software;
    }

    const auto& hwSupportMap = supportIterator->second.Hw;
    auto hwSupportIterator = hwSupportMap.find(deviceGeneration);
    if (hwSupportIterator == hwSupportMap.end())
    {
        return SubmodelType::Software;
    }

    auto isSupportedByHardware = hwSupportIterator->second;
    if (!isSupportedByHardware)
    {
        return SubmodelType::Software;
    }

    if (hwCaps.IsLayerSupported(layer.Operation))
    {
        return SubmodelType::Hardware;
    }

    if (INTEL_GMM == layer.Operation && hwCaps.HasFeature(LegacyGMM))
    {
        return SubmodelType::GMMHardware;
    }

    return SubmodelType::Software;
}

void CompiledModel::createSubmodels(const HardwareCapabilities& hwCaps)
{
    auto const deviceVersion = hwCaps.GetDeviceVersion();
    auto& deviceSubmodels = submodels[deviceVersion];

    auto layerIndex = uint32_t{ 0 };

    auto submodelType = getSubmodelType(hwCaps, layerIndex);

    deviceSubmodels.emplace_back(std::make_unique<SubModel>(submodelType, 0));
    auto currentSubmodel = deviceSubmodels.back().get();
    Expect::NotNull(currentSubmodel);
    layerIndex++;

    for (; layerIndex < LayerCount; ++layerIndex)
    {
        submodelType = getSubmodelType(hwCaps, layerIndex);

        auto doSplit = false;

        if (GMMHardware == submodelType
            || currentSubmodel->Type != submodelType)
        {
            doSplit = true;
        }
        else if (Hardware == submodelType
            && currentSubmodel->GetLayerCount() == hwCaps.GetMaximumLayerCount())
        {
            doSplit = true;
        }

        if (doSplit)
        {
            deviceSubmodels.emplace_back(
                std::make_unique<SubModel>(submodelType, layerIndex));
            currentSubmodel = deviceSubmodels.back().get();
            Expect::NotNull(currentSubmodel);
        }
        else
        {
            currentSubmodel->AddLayer();
        }
    }
}
