/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Device.h"

#include "ActiveList.h"
#include "Expect.h"
#include "GnaException.h"
#include "RequestConfiguration.h"

#include <cstdint>
#include <memory>

using namespace GNA;

uint32_t Device::modelIdSequence = 0;

Device::Device(std::unique_ptr<HardwareCapabilities>&& hardwareCapabilitiesIn) :
    hardwareCapabilities{ std::move(hardwareCapabilitiesIn) }
{
    accelerationDetector.SetHardwareAcceleration(hardwareCapabilities->IsHardwareSupported());
    accelerationDetector.PrintAllAccelerationModes();
}

DeviceVersion Device::GetVersion() const
{
    return hardwareCapabilities->GetHardwareDeviceVersion();
}

uint32_t Device::GetNumberOfThreads() const
{
    return requestHandler.GetNumberOfThreads();
}

void Device::SetNumberOfThreads(uint32_t threadCount)
{
    requestHandler.ChangeNumberOfThreads(threadCount);
}

uint32_t Device::StoreModel(std::unique_ptr<CompiledModel> && compiledModel)
{
    if (!compiledModel)
    {
        throw GnaException(Gna2StatusResourceAllocationError);
    }

    auto modelId = modelIdSequence++;

    models.emplace(modelId, std::move(compiledModel));
    return modelId;
}

void Device::AttachBuffer(uint32_t configId,
    uint32_t operandIndex, uint32_t layerIndex, void *address)
{
    Expect::NotNull(address);

    requestBuilder.AttachBuffer(configId, operandIndex, layerIndex, address);
}

void Device::CreateConfiguration(uint32_t modelId, uint32_t *configId)
{
    auto &model = *models.at(modelId);
    requestBuilder.CreateConfiguration(model, configId,
        *hardwareCapabilities);
}

void Device::ReleaseConfiguration(uint32_t configId)
{
    requestBuilder.ReleaseConfiguration(configId);
}

bool Device::IsVersionConsistent(DeviceVersion deviceVersion) const
{
    return hardwareCapabilities->GetDeviceVersion() == deviceVersion;
}

void Device::EnforceAcceleration(uint32_t configId, Gna2AccelerationMode accelerationMode)
{
    auto& requestConfiguration = requestBuilder.GetConfiguration(configId);
    requestConfiguration.EnforceAcceleration(accelerationMode);
}

void Device::AttachActiveList(uint32_t configId, uint32_t layerIndex,
    uint32_t indicesCount, const uint32_t* const indices)
{
    Expect::NotNull(indices);

    const auto activeList = ActiveList{ indicesCount, indices };
    requestBuilder.AttachActiveList(configId, layerIndex, activeList);
}

bool Device::HasRequestConfigId(uint32_t requestConfigId) const
{
    return requestBuilder.HasConfiguration(requestConfigId);
}

bool Device::HasRequestId(uint32_t requestId) const
{
    return requestHandler.HasRequest(requestId);
}

CompiledModel const& Device::GetModel(uint32_t modelId)
{
    return *models.at(modelId);
}

void Device::ReleaseModel(uint32_t const modelId)
{
    models.erase(modelId);
}

void Device::PropagateRequest(uint32_t configId, uint32_t *requestId)
{
    Expect::NotNull(requestId);

    auto request = requestBuilder.CreateRequest(configId);
    requestHandler.Enqueue(requestId, std::move(request));
}

Gna2Status Device::WaitForRequest(uint32_t requestId, uint32_t milliseconds)
{
    return requestHandler.WaitFor(requestId, milliseconds);
}

void Device::Stop()
{
    requestHandler.StopRequests();
}

bool Device::HasModel(uint32_t modelId) const
{
    return models.count(modelId) > 0;
}

void Device::AssignProfilerConfigToRequestConfig(uint32_t requestConfigId, ProfilerConfiguration& profilerConfiguration)
{
    auto& requestConfiguration = requestBuilder.GetConfiguration(requestConfigId);
    requestConfiguration.AssignProfilerConfig(&profilerConfiguration);
}