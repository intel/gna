/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "AccelerationDetector.h"
#include "MemoryContainer.h"
#include "SoftwareModel.h"
#include "Validator.h"

#include "gna2-capability-impl.h"
#include "gna2-common-api.h"
#include "gna2-model-api.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct KernelBuffers;

namespace GNA
{
class HardwareCapabilities;
class Layer;
class Memory;
class RequestConfiguration;
class RequestProfiler;

class CompiledModel
{
public:
    virtual ~CompiledModel() = default;
    CompiledModel(const CompiledModel&) = delete;
    CompiledModel(CompiledModel&&) = delete;
    CompiledModel& operator=(const CompiledModel&) = delete;
    CompiledModel& operator=(CompiledModel&&) = delete;

    std::vector<std::unique_ptr<Layer>> const & GetLayers() const
    {
        return GetSoftwareModel().GetLayers();
    }

    uint32_t GetScratchpadSize() const;

    Layer const & GetLayer(uint32_t layerIndex) const
    {
        return GetSoftwareModel().GetLayer(layerIndex);
    }

    uint32_t GetMaximumOperandSize(uint32_t operandIndex);

    void VerifyBufferAndStoreMemory(const void *buffer, size_t bufferSize, uint32_t alignment);

    uint32_t GetSize() const
    {
        return allocations.GetMemorySize();
    }

    Memory const * GetMemoryIfNotPartOfModel(const void *buffer, size_t bufferSize) const;

    auto const & GetBufferConfigValidator() const
    {
        return  GetSoftwareModel().GetBufferConfigValidator();
    }

    Gna2Status Score(
        RequestConfiguration& config,
        RequestProfiler &profiler,
        KernelBuffers *buffers);

    MemoryContainer const & GetAllocations() const
    {
        return allocations;
    }

    void InvalidateRequestConfig(uint32_t configId) const;

    void ValidateBuffer(MemoryContainer const & requestAllocations, Memory const & memory) const;

    bool IsHardwareEnforcedModeValid();

    const uint32_t LayerCount;
    const uint32_t GmmCount;

protected:
    CompiledModel(
        const ApiModel & model,
        const AccelerationDetector& detectorIn,
        const HardwareCapabilities& hwCapabilitiesIn,
        Gna2DeviceVersion softwareModelVersion);

    BaseValidator makeValidator(Gna2DeviceGeneration generation);

    static uint32_t GetNumberOfOperations(const Gna2Model& model, Gna2DeviceVersion softwareModelVersion)
    {
        HardwareCapabilities::ValidateOperationCount(model.NumberOfOperations, softwareModelVersion);
        return model.NumberOfOperations;
    }

    static Gna2Operation* GetFirstOperation(const Gna2Model& model)
    {
        return model.Operations;
    }

    static bool isGmmOperation(const Gna2Operation& operation)
    {
        return operation.Type == Gna2OperationTypeGmm;
    }

    template<class T>
    static uint32_t getGmmCount(const T* firstOperation, uint32_t numberOfOperations)
    {
        uint32_t gmmCount = 0;
        for (uint32_t i = 0; i < numberOfOperations; i++)
        {
            if (isGmmOperation(firstOperation[i]))
            {
                ++gmmCount;
            }
        }
        return gmmCount;
    }

    Memory const & getMemoryFromDeviceAllocations(const void *buffer, size_t bufferSize) const;

    const AccelerationDetector& detector;

    const HardwareCapabilities& hwCapabilities;

    MemoryContainer allocations;

    const ApiModel & apiModel;

    virtual SoftwareModel & GetSoftwareModel()
    {
        return softwareModel;
    }

    virtual SoftwareModel const & GetSoftwareModel() const
    {
        return softwareModel;
    }

    // Software only model, built with latest/relaxed limitations
    // used for Software scoring only
    // not used with hardware model (actually some common properties may be used)
    SoftwareModel softwareModel;

private:
    virtual void score(ScoreContext & context) = 0;

    virtual void invalidateRequestConfig(uint32_t configId) const = 0;

    virtual void validateBuffer(MemoryContainer const & requestAllocations, Memory const & memory) const = 0;

    virtual bool isFullyHardwareCompatible() = 0;
};

}
