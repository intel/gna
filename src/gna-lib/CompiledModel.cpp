/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#define NOMINMAX 1

#include "CompiledModel.h"

#include "DeviceManager.h"
#include "Expect.h"
#include "gna2-capability-impl.h"
#include "gna2-memory-impl.h"
#include "GnaException.h"
#include "HardwareCapabilities.h"
#include "Layer.h"
#include "Logger.h"
#include "Memory.h"
#include "Request.h"
#include "RequestConfiguration.h"

#include "gna2-device-api.h"

using namespace GNA;

uint32_t CompiledModel::GetScratchpadSize() const
{
    auto scratchPadSize = 0u;
    for (auto const & layer : GetLayers())
    {
        auto const scratchPad = layer->TryGetOperand(ScratchpadOperandIndex);
        if (scratchPad)
        {
            scratchPadSize = std::max(scratchPadSize, scratchPad->Size);
        }
    }
    scratchPadSize = RoundUp(scratchPadSize, Memory::GNA_BUFFER_ALIGNMENT);
    return scratchPadSize;
}

uint32_t CompiledModel::GetMaximumOperandSize(uint32_t operandIndex)
{
    return GetSoftwareModel().GetMaximumOperandSize(operandIndex);
}

void CompiledModel::VerifyBufferAndStoreMemory(const void *buffer, size_t bufferSize, uint32_t alignment)
{
    ModelErrorHelper::ExpectNotNull(buffer);
    ModelErrorHelper::ExpectBufferAligned(buffer, alignment);
    try
    {
        if (!allocations.Contains(buffer, bufferSize))
        {
            auto const & memory = getMemoryFromDeviceAllocations(buffer, bufferSize);
            allocations.Emplace(memory);
        }
    }
    catch (GnaException&)
    {
        throw GnaModelErrorException(Gna2ItemTypeOperandData,
            Gna2ErrorTypeArgumentInvalid, reinterpret_cast<int64_t>(buffer));
    }

}


Memory const * CompiledModel::GetMemoryIfNotPartOfModel(const void *buffer, size_t bufferSize) const
{
    if (allocations.Contains(buffer, bufferSize))
    {
        return nullptr;
    }

    return &getMemoryFromDeviceAllocations(buffer, bufferSize);
}

Gna2Status CompiledModel::Score(
    RequestConfiguration& config,
    RequestProfiler &profiler,
    KernelBuffers *buffers)
{
    auto context = ScoreContext{ 0, LayerCount, config, profiler, buffers };
    try
    {
        profiler.Measure(Gna2InstrumentationPointLibProcessing);
        score(context);
        profiler.Measure(Gna2InstrumentationPointLibCompletion);
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
    return context.saturationCount > 0 ? Gna2StatusWarningArithmeticSaturation : Gna2StatusSuccess;
}
void CompiledModel::InvalidateRequestConfig(uint32_t configId) const
{
    invalidateRequestConfig(configId);
}

void CompiledModel::ValidateBuffer(MemoryContainer const & requestAllocations, Memory const & memory) const
{
    validateBuffer(requestAllocations, memory);
}

bool CompiledModel::IsHardwareEnforcedModeValid()
{
    return isFullyHardwareCompatible();
}

CompiledModel::CompiledModel(const ApiModel & model, const AccelerationDetector& detectorIn, const HardwareCapabilities& hwCapabilitiesIn,
    Gna2DeviceVersion softwareModelVersion) :
    LayerCount{ GetNumberOfOperations(model, softwareModelVersion) },
    GmmCount{ getGmmCount(GetFirstOperation(model), LayerCount) },
    detector{ detectorIn },
    hwCapabilities{ hwCapabilitiesIn },
    apiModel{ model },
    softwareModel
    {
        apiModel,
        makeValidator(HardwareCapabilities::GetDeviceGeneration(softwareModelVersion)),
        detector.GetSupportedCpuAccelerations()
    }
{
}

BaseValidator CompiledModel::makeValidator(Gna2DeviceGeneration generation)
{
    return BaseValidator
    {
        generation,
        ValidBoundariesFunctor {
            [this](const void *buffer, size_t bufferSize, uint32_t alignment)
            {
                VerifyBufferAndStoreMemory(buffer, bufferSize, alignment);
            }
        }
    };
}

Memory const & CompiledModel::getMemoryFromDeviceAllocations(const void *buffer, size_t bufferSize) const
{
    const auto& allAllocations = DeviceManager::Get().GetAllAllocated();

    for (auto const & memory : allAllocations)
    {
        Expect::NotNull(memory, Gna2StatusXnnErrorInvalidBuffer);

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
