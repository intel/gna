/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "HardwareModelScorable.h"

#include "CompiledModel.h"
#include "DriverInterface.h"
#include "Expect.h"
#include "GnaException.h"
#include "gna2-memory-impl.h"
#include "HardwareCapabilities.h"
#include "Memory.h"
#include "MemoryContainer.h"
#include "RequestConfiguration.h"
#include "SoftwareModel.h"

#include <utility>

using namespace GNA;

HardwareModelScorable::HardwareModelScorable(CompiledModel const & softwareModel,
    DriverInterface &ddi, const HardwareCapabilities& hwCapsIn, const std::vector<std::unique_ptr<SubModel>>& subModelsIn) :
    HardwareModel(softwareModel, hwCapsIn),
    driverInterface{ ddi },
    subModels{ subModelsIn }
{
    Build();
}

uint32_t HardwareModelScorable::GetBufferOffsetForConfiguration(
    const BaseAddress& address,
    const RequestConfiguration& requestConfiguration) const
{
    auto offset = HardwareModel::GetBufferOffset(address).Offset;
    if (offset != 0)
    {
        return offset;
    }

    auto const modelSize = allocations.GetMemorySizeAlignedToPage();
    offset = requestConfiguration.GetAllocations().GetBufferOffset(address, MemoryBufferAlignment, modelSize);
    Expect::GtZero(offset, Gna2StatusMemoryBufferInvalid);
    return offset;
}

void HardwareModelScorable::InvalidateConfig(uint32_t configId)
{
    if (hardwareRequests.find(configId) != hardwareRequests.end())
    {
        hardwareRequests[configId]->Invalidate();
    }
}

void HardwareModelScorable::Score(ScoreContext & context)
{
    if (context.layerIndex + context.layerCount > hardwareLayers.size())
    {
        throw GnaException(Gna2StatusXnnErrorNetLyrNo);
    }
    for (auto i = context.layerIndex; i < context.layerIndex + context.layerCount; i++)
    {
        Expect::NotNull(TryGetLayer(i), Gna2StatusXnnErrorNetLyrNo);
    }

    auto operationMode = xNN;

    auto const & layer = model.GetLayer(context.layerIndex);
    if (layer.Operation == INTEL_GMM
        && !hwCapabilities.IsOperationSupported(layer.Operation)
        && hwCapabilities.HasFeature(LegacyGMM))
    {
        Expect::InRange(context.layerCount, 1u, Gna2StatusXnnErrorNetLyrNo);
        operationMode = GMM;
    }

    hwCapabilities.ValidateOperationCount(context.layerCount);

    SoftwareModel::LogAcceleration(AccelerationMode{ Gna2AccelerationModeHardware });
    SoftwareModel::LogOperationMode(operationMode);

    auto configId = context.requestConfiguration.Id;
    HardwareRequest *hwRequest;
    {
        std::lock_guard<std::mutex> lockGuard(hardwareRequestsLock);
        if (hardwareRequests.find(configId) == hardwareRequests.end())
        {
            auto const inserted = hardwareRequests.emplace(
                configId,
                std::make_unique<HardwareRequest>(*this, context.requestConfiguration, allocations));
            hwRequest = inserted.first->second.get();
        }
        else
        {
            hwRequest = hardwareRequests.at(configId).get();
        }
    }
    hwRequest->Update(context.layerIndex, context.layerCount, operationMode);

    context.profiler.Measure(Gna2InstrumentationPointLibExecution);

    auto const result = driverInterface.Submit(*hwRequest, context.profiler);
    context.profiler.AddResults(Gna2InstrumentationPointDrvPreprocessing, result.driverPerf.Preprocessing);
    context.profiler.AddResults(Gna2InstrumentationPointDrvProcessing, result.driverPerf.Processing);
    context.profiler.AddResults(Gna2InstrumentationPointDrvDeviceRequestCompleted, result.driverPerf.DeviceRequestCompleted);
    context.profiler.AddResults(Gna2InstrumentationPointDrvCompletion, result.driverPerf.Completion);

    context.profiler.AddResults(Gna2InstrumentationPointHwTotalCycles, result.hardwarePerf.total);
    context.profiler.AddResults(Gna2InstrumentationPointHwStallCycles, result.hardwarePerf.stall);

    if (result.status != Gna2StatusSuccess && result.status != Gna2StatusWarningArithmeticSaturation)
    {
        throw GnaException(result.status);
    }

    context.saturationCount += Gna2StatusWarningArithmeticSaturation == result.status ? 1 : 0;
}

void HardwareModelScorable::ValidateConfigBuffer(MemoryContainer const & requestAllocations,
    Memory const & bufferMemory) const
{
    auto configModelSize = allocations.GetMemorySizeAlignedToPage();
    configModelSize += requestAllocations.GetMemorySizeAlignedToPage();
    configModelSize += RoundUp(bufferMemory.GetSize(), MemoryBufferAlignment);

    Expect::InRange(configModelSize, HardwareCapabilities::MaximumModelSize,
        Gna2StatusMemoryTotalSizeExceeded);
}

void HardwareModelScorable::prepareAllocationsAndModel()
{
    HardwareModel::prepareAllocationsAndModel();
    ldMemory->Map(driverInterface);
    getHwOffsetFunction = [this](const BaseAddress& buffer) { return GetBufferOffset(buffer); };
}

bool HardwareModelScorable::IsSoftwareLayer(uint32_t layerIndex) const
{
    return SubModel::IsSoftwareLayer(layerIndex, subModels);
}

std::unique_ptr<Memory> HardwareModelScorable::allocLD(uint32_t ldMemorySize, uint32_t ldSize)
{
    return std::make_unique<Memory>(driverInterface.MemoryCreate(ldMemorySize, ldSize));
}
