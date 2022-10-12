/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "BufferConfigValidator.h"
#include "LayerConfiguration.h"
#include "MemoryContainer.h"
#include "ProfilerConfiguration.h"
#include "Tensor.h"

#include "gna2-common-impl.h"
#include "gna2-inference-api.h"
#include "gna2-inference-impl.h"

#include <map>
#include <memory>
#include <cstdint>

namespace GNA
{

class CompiledModel;

class Memory;

struct ActiveList;

/*
** RequestConfiguration is a bunch of request buffers
** sent to GNA kernel driver as part of WRITE request
**
 */
class RequestConfiguration
{
public:
    RequestConfiguration(CompiledModel& model, uint32_t configId, const HardwareCapabilities & hardwareCapabilitiesIn);

    ~RequestConfiguration() = default;

    void AddBuffer(uint32_t operandIndex, uint32_t layerIndex, void *address);

    void AddActiveList(uint32_t layerIndex, const ActiveList& activeList);

    void EnforceAcceleration(Gna2AccelerationMode accelerationMode);

    DeviceVersion GetConsistentDevice() const;

    void AssignProfilerConfig(ProfilerConfiguration* config);

    ProfilerConfiguration* GetProfilerConfiguration() const
    {
        return profilerConfiguration;
    }

    uint8_t GetHwInstrumentationMode() const;

    MemoryContainer const & GetAllocations() const
    {
        return allocations;
    }

    void UpdateConsistency(DeviceVersion consistentVersion);

    void Validate() const
    {
        bufferConfigValidator.validate();
    }

    CompiledModel & Model;

    const uint32_t Id;

    std::map<uint32_t, std::unique_ptr<LayerConfiguration>> LayerConfigurations;

    uint32_t ActiveListCount = 0;

    // Number of elements in buffer per input precision and per grouping
    uint32_t const * BufferElementCount = nullptr;
    uint32_t const * BufferElementCountFor3_0 = nullptr;

    AccelerationMode Acceleration = Gna2AccelerationModeAuto;

private:
    struct AddBufferContext
    {
        AddBufferContext(CompiledModel & model, uint32_t operandIndexIn, uint32_t layerIndexIn, void *addressIn);

        Layer const * SoftwareLayer;
        Tensor const * Operand;
        uint32_t OperandIndex;
        uint32_t LayerIndex;
        void * Address;
        uint32_t Size;
    };

    void storeAllocationIfNew(void const * buffer, uint32_t bufferSize);

    void applyBufferForSingleLayer(AddBufferContext & context);

    void addBufferForMultipleLayers(AddBufferContext & context);

    void addBufferForSingleLayer(AddBufferContext & context);

    LayerConfiguration & getLayerConfiguration(uint32_t layerIndex);

    void updateMissingBufferForSingleLayer(AddBufferContext & context);

    ProfilerConfiguration* profilerConfiguration = nullptr;

    MemoryContainer allocations;

    const HardwareCapabilities & hardwareCapabilities;

    // Per request copy of config from software model
    BufferConfigValidator bufferConfigValidator;
};

}
