/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "AccelerationDetector.h"
#include "HardwareModelScorable.h"
#include "MemoryContainer.h"
#include "SoftwareModel.h"
#include "SubModel.h"
#include "Validator.h"

#include "gna2-common-api.h"
#include "gna2-common-impl.h"
#include "gna2-model-api.h"

#include "common.h"
#include "gna-api.h"
#include "gna-api-types-xnn.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct KernelBuffers;

namespace GNA
{
class DriverInterface;
class HardwareCapabilities;
class Layer;
class Memory;
class RequestConfiguration;
struct LayerConfiguration;
class RequestProfiler;

class CompiledModel
{
public:
    template<class T>
    CompiledModel(
        const T & model,
        const AccelerationDetector& detectorIn,
        const HardwareCapabilities& hwCapabilitiesIn) :
        LayerCount{ GetNumberOfOperations(model) },
        GmmCount{ getGmmCount(GetFirstOperation(model), LayerCount) },
        detector{ detectorIn },
        hwCapabilities{ hwCapabilitiesIn },
        softwareModel
    {
        model,
        makeValidator(),
        detector.GetSupportedCpuAccelerations()
    }
    {}

    virtual ~CompiledModel() = default;
    CompiledModel(const CompiledModel &) = delete;
    CompiledModel& operator=(const CompiledModel&) = delete;

    void BuildHardwareModel(DriverInterface &ddi);

    std::vector<std::unique_ptr<Layer>> const & GetLayers() const
    {
        return softwareModel.GetLayers();
    }

    Layer const & GetLayer(uint32_t layerIndex) const
    {
        return softwareModel.GetLayer(layerIndex);
    }

    uint32_t GetMaximumOperandSize(uint32_t operandIndex);

    void VerifyBufferAndStoreMemory(const void *buffer, size_t bufferSize);

    uint32_t GetSize() const
    {
        return allocations.GetMemorySize();
    }

    void CopyData(void *address, size_t size) const;

    void InvalidateHardwareRequestConfig(uint32_t configId) const;

    Memory const * GetMemoryIfNotPartOfModel(const void *buffer, size_t bufferSize) const;

    Gna2Status Score(
        RequestConfiguration& config,
        RequestProfiler *profiler,
        KernelBuffers *buffers);

    void ValidateBuffer(MemoryContainer const & requestAllocations, Memory const & memory) const;

    MemoryContainer const & GetAllocations() const
    {
        return allocations;
    }

    bool IsHardwareEnforcedModeValid();
    bool IsFullyHardwareCompatible(const HardwareCapabilities& targetDevice);

    const uint32_t LayerCount;
    const uint32_t GmmCount;

protected:
    std::unique_ptr<HardwareModelScorable> hardwareModel;

private:

    enum AccelerationType
    {
        Unsupported,
        Auto,
        EnforcedSoftware,
    };

    const std::vector<std::unique_ptr<SubModel>>&
        getSubmodels(const HardwareCapabilities& hwCaps);

    void createSubmodels(const HardwareCapabilities& hwCaps);

    SubmodelType getSubmodelType(
        const HardwareCapabilities &hwCaps, uint32_t layerIndex) const;

    AccelerationType getEffectiveAccelerationMode(RequestConfiguration& config);

    uint32_t scoreAllSubModels(RequestConfiguration& config,
        RequestProfiler *profiler, KernelBuffers *buffers);

    BaseValidator makeValidator();
    static uint32_t GetNumberOfOperations(const Gna2Model& model)
    {
        return model.NumberOfOperations;
    }
    static Gna2Operation* GetFirstOperation(const Gna2Model& model)
    {
        return model.Operations;
    }
    static bool isGmmOperation(const nn_layer& layer)
    {
        return layer.operation == INTEL_GMM;
    }
    static bool isGmmOperation(const Gna2Operation& operation)
    {
        return operation.Type == Gna2OperationTypeGmm;
    }

    template<class T>
    uint32_t getGmmCount(const T* firstOperation, uint32_t numberOfOperations)
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

    Memory const & getMemoryFromDeviceAllocations(const void *buffer, const size_t bufferSize) const;

    const AccelerationDetector& detector;
    const HardwareCapabilities& hwCapabilities;
    MemoryContainer allocations;
    SoftwareModel softwareModel;
    std::map<DeviceVersion,
        std::vector<std::unique_ptr<SubModel>>> submodels;

};

}
