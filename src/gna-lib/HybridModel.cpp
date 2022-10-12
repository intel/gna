/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#define NOMINMAX 1

#include "HybridModel.h"

#include "DeviceManager.h"
#include "Expect.h"
#include "GnaException.h"
#include "HardwareCapabilities.h"
#include "Layer.h"
#include "Logger.h"
#include "Memory.h"
#include "RequestConfiguration.h"
#include "SubModel.h"

using namespace GNA;

HybridModel::HybridModel(const ApiModel& model, const AccelerationDetector& detectorIn,
    const HardwareCapabilities& hwCapabilitiesIn, DriverInterface& ddi) :
    CompiledModel{ model, detectorIn, hwCapabilitiesIn, Gna2DeviceVersionSoftwareEmulation }
{
    // try build hw model but do not throw on error, store error instead to allow sw scoring when no HW is present or model is not compatible
    try
    {
        BuildHardwareModel(ddi);
    }
    catch (GnaModelErrorException& exception)
    {
        ModelErrorHelper::SaveLastError(exception.GetModelError());
        Log->Warning("Building Software and Hardware model for present device failed, error saved.\n");
    }
    catch (GnaException& exception)
    {
        if (!Gna2StatusIsSuccessful(exception.GetStatus()) &&
            Gna2StatusResourceAllocationError != exception.GetStatus())
        {
            ModelErrorHelper::SaveLastError(ModelError(exception.GetStatus()));
            Log->Warning("Building Software and Hardware model for present device failed with unknown reason, error saved.\n");
        }
        else
        {
            throw; // handle as non "gna runtime" exception, not model related
        }
    }
}

bool HybridModel::verifyFullyHardwareCompatible()
{
    const auto& deviceSubModels = getSubModels();
    return std::none_of(deviceSubModels.begin(), deviceSubModels.end(),
        [](auto && subModel) {return subModel->Type == Software; });
}

void HybridModel::BuildHardwareModel(DriverInterface &ddi)
{
    if (!hwCapabilities.IsHardwareSupported())
    {
        return;
    }

    const auto& deviceSubModels = getSubModels();
    const auto noHardwareCompliantOperation = std::none_of(deviceSubModels.begin(), deviceSubModels.end(),
        [](auto && subModel) { return subModel->Type != Software; });
    if (noHardwareCompliantOperation)
    {
        const auto error = ModelError(Gna2ErrorTypeNoHardwareCompliantOperation, 0, ModelItem());
        throw GnaModelErrorException(error);
    }

    softwareModelForPresentDevice = std::make_unique<SoftwareModel>(apiModel,
        makeValidator(HardwareCapabilities::GetDeviceGeneration(Gna2DeviceVersionSoftwareEmulation)),
        makeValidator(hwCapabilities.GetDeviceGeneration()),
        detector.GetSupportedCpuAccelerations(),
        subModels.at(hwCapabilities.GetDeviceVersion()));
    Expect::NotNull(softwareModelForPresentDevice, Gna2StatusResourceAllocationError);

    hardwareModel = std::make_unique<HardwareModelScorable>(*this, ddi, hwCapabilities, deviceSubModels);
    Expect::NotNull(hardwareModel, Gna2StatusResourceAllocationError);

    fullyHardwareCompatible = verifyFullyHardwareCompatible();
}

void HybridModel::score(ScoreContext& context)
{
    if (shouldUseSoftwareMode(context.requestConfiguration))
    {
        context.requestConfiguration.UpdateConsistency(getSoftwareConsistencyDeviceVersion());
        if (softwareModelForPresentDevice)
        {
            softwareModelForPresentDevice->Score(context);
        }
        else
        {
            softwareModel.Score(context);
        }
    }
    else
    {
        for (const auto& subModel : getSubModels())
        {
            context.Update(subModel.get());
            ScoreSubModel(context);
        }
    }
}

void HybridModel::ScoreSubModel(ScoreContext & context)
{
    switch (context.subModelType)
    {
    case Hardware:
    case GMMHardware:
        return ScoreHwSubModel(context);
    case Software:
        context.requestConfiguration.UpdateConsistency(Gna2DeviceVersionSoftwareEmulation);
        return softwareModelForPresentDevice->Score(context);
    }
}

void HybridModel::ScoreHwSubModel(ScoreContext & context)
{
    try
    {
        hardwareModel->Score(context);
    }
    catch (GnaException & e)
    {
        if (context.requestConfiguration.Acceleration.IsSoftwareFallbackEnabled() && e.GetStatus() == Gna2StatusDeviceQueueError)
        {
            // fallback to Software mode with HW compatible model
            context.requestConfiguration.UpdateConsistency(hwCapabilities.GetDeviceVersion());
            softwareModelForPresentDevice->Score(context);
        }
        else //unrecoverable exception
        {
            throw;
        }
    }
}

const std::vector<std::unique_ptr<SubModel>>& HybridModel::getSubModels()
{
    if (subModels.find(hwCapabilities.GetDeviceVersion()) == subModels.end())
    {
        createSubModels(hwCapabilities);
    }

    return subModels.at(hwCapabilities.GetDeviceVersion());
}

SubModelType HybridModel::getSubModelType(
    const HardwareCapabilities &hwCaps, uint32_t layerIndex) const
{
    auto const & layer = GetLayer(layerIndex);
    auto const dataConfig = layer.GetDataMode();
    auto const deviceGeneration = hwCaps.GetDeviceGeneration();
    auto const isSupported = DataConfig::IsOperationSupported(layer.Operation, dataConfig, deviceGeneration);

    if (!isSupported)
    {
        return Software;
    }

    if (hwCaps.IsOperationSupported(layer.Operation))
    {
        return Hardware;
    }

    if (INTEL_GMM == layer.Operation && hwCaps.HasFeature(LegacyGMM))
    {
        return GMMHardware;
    }

    return Software;
}

void HybridModel::createSubModels(const HardwareCapabilities& hwCaps)
{
    auto const deviceVersion = hwCaps.GetDeviceVersion();
    auto& deviceSubModels = subModels[deviceVersion];

    auto layerIndex = uint32_t{ 0 };

    auto subModelType = getSubModelType(hwCaps, layerIndex);

    deviceSubModels.emplace_back(std::make_unique<SubModel>(subModelType, 0));
    auto currentSubModel = deviceSubModels.back().get();
    Expect::NotNull(currentSubModel);
    layerIndex++;

    for (; layerIndex < LayerCount; ++layerIndex)
    {
        subModelType = getSubModelType(hwCaps, layerIndex);

        if (shouldSplit(*currentSubModel, subModelType, hwCaps))
        {
            deviceSubModels.emplace_back(
                std::make_unique<SubModel>(subModelType, layerIndex));
            currentSubModel = deviceSubModels.back().get();
            Expect::NotNull(currentSubModel);
        }
        else
        {
            currentSubModel->AddLayer();
        }
    }
}

bool HybridModel::shouldSplit(SubModel& currentSubModel, SubModelType nextSubModelType, const HardwareCapabilities& hwCaps)
{
    if (currentSubModel.Type != nextSubModelType)
    {
        return true;
    }
    if (GMMHardware == nextSubModelType)
    {
        return true;
    }
    if (Hardware == nextSubModelType && currentSubModel.GetLayerCount() == hwCaps.GetMaximumLayerCount())
    {
        return true;
    }

    return false;
}

void HybridModel::invalidateRequestConfig(uint32_t configId) const
{
    if (hardwareModel)
    {
        hardwareModel->InvalidateConfig(configId);
    }
}

void HybridModel::validateBuffer(MemoryContainer const & requestAllocations, Memory const & memory) const
{
    if (hardwareModel)
    {
        hardwareModel->ValidateConfigBuffer(requestAllocations, memory);
    }
}

bool HybridModel::isFullyHardwareCompatible()
{
    return fullyHardwareCompatible;
}

bool HybridModel::shouldUseSoftwareMode(RequestConfiguration& config) const
{
    const auto isSoftwareEffective = config.Acceleration.IsSoftwareEnforced() ||
        (config.Acceleration.GetMode() == Gna2AccelerationModeAuto && !hardwareModel);
    return isSoftwareEffective;
}

DeviceVersion HybridModel::getSoftwareConsistencyDeviceVersion() const
{
    if (fullyHardwareCompatible)
    {
        return hwCapabilities.GetDeviceVersion();
    }
    // only SoftwareEmulation consistency is possible
    return Gna2DeviceVersionSoftwareEmulation;
}
