/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "AccelerationDetector.h"
#include "CompiledModel.h"
#include "DriverInterface.h"
#include "HardwareCapabilities.h"
#include "Memory.h"
#include "RequestBuilder.h"
#include "RequestHandler.h"

#include "gna-api.h"
#include "gna2-instrumentation-api.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct Gna2ModelSueCreekHeader;

namespace GNA
{

class Device
{
public:
    Device(uint32_t deviceIndex, uint32_t threadCount = 1);
    Device(const Device &) = delete;
    Device& operator=(const Device&) = delete;
    virtual ~Device() = default;

    DeviceVersion GetVersion() const;

    uint32_t GetNumberOfThreads() const;

    void SetNumberOfThreads(uint32_t threadCount);

    template<class T>
    uint32_t LoadModel(const T& model)
    {
        auto compiledModel = std::make_unique<CompiledModel>(
            model, accelerationDetector, hardwareCapabilities);

        if (!compiledModel)
        {
            throw GnaException(Gna2StatusResourceAllocationError);
        }

        auto modelId = modelIdSequence++;

        compiledModel->BuildHardwareModel(*driverInterface);
        models.emplace(modelId, std::move(compiledModel));
        return modelId;
    }

    void ReleaseModel(uint32_t const modelId);

    void AttachBuffer(uint32_t configId, uint32_t operandIndex, uint32_t layerIndex, void *address);

    void CreateConfiguration(uint32_t modelId, uint32_t *configId);

    void ReleaseConfiguration(uint32_t configId);

    void EnableHardwareConsistency(uint32_t configId, DeviceVersion deviceVersion);

    void EnforceAcceleration(uint32_t configId, Gna2AccelerationMode accel);

    void AttachActiveList(uint32_t configId, uint32_t layerIndex, uint32_t indicesCount, const uint32_t* const indices);

    void PropagateRequest(uint32_t configId, uint32_t *requestId);

    Gna2Status WaitForRequest(uint32_t requestId, uint32_t milliseconds);

    void Stop();

    void* Dump(uint32_t modelId, Gna2ModelSueCreekHeader* modelHeader, Gna2Status* status,
            Gna2UserAllocator customAlloc);

    void DumpLdNoMMu(uint32_t modelId, Gna2UserAllocator customAlloc,
        void *& exportData, uint32_t & exportDataSize);

    uint32_t CreateProfilerConfiguration(std::vector<Gna2InstrumentationPoint>&& selectedInstrumentationPoints, uint64_t* results);

    void ReleaseProfilerConfiguration(uint32_t configId);

    void AssignProfilerConfigToRequestConfig(uint32_t instrumentationConfigId, uint32_t requestConfigId);

    void SetInstrumentationUnit(uint32_t configId, Gna2InstrumentationUnit instrumentationUnit);

    void SetHardwareInstrumentation(uint32_t configId, enum Gna2InstrumentationMode instrumentationMode);

    bool HasModel(uint32_t modelId) const;

    bool HasMemory(void * memory) const;

    bool HasRequestConfigId(uint32_t requestConfigId) const;

    bool HasRequestId(uint32_t requestId) const;

    void MapMemory(Memory& memoryObject);

    void UnMapMemory(Memory & memoryObject);

protected:
    static const std::map<const gna_device_generation, const DeviceVersion> deviceDictionary;

    uint32_t id;

    static uint32_t modelIdSequence;

    std::unique_ptr<DriverInterface> driverInterface;

    HardwareCapabilities hardwareCapabilities;

    AccelerationDetector accelerationDetector;

    RequestBuilder requestBuilder;

    RequestHandler requestHandler;

    std::map<uint32_t, std::unique_ptr<CompiledModel>> models;

private:
    uint32_t numberOfThreads;
};
}
