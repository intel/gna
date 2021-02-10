/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "gna2-inference-impl.h"

#include "ApiWrapper.h"
#include "Logger.h"
#include "DeviceManager.h"
#include "Macros.h"
#include "ModelWrapper.h"

#include "gna2-common-impl.h"

#include <stdint.h>
#include <vector>


using namespace GNA;

GNA2_API enum Gna2Status Gna2RequestConfigCreate(
    uint32_t modelId,
    uint32_t * requestConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDeviceForModel(modelId);
        device.CreateConfiguration(modelId, requestConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigSetOperandBuffer(
    uint32_t requestConfigId,
    uint32_t operationIndex,
    uint32_t operandIndex,
    void * address)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(requestConfigId);
        device.AttachBuffer(requestConfigId, operandIndex, operationIndex, address);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigEnableActiveList(
    uint32_t requestConfigId,
    uint32_t operationIndex,
    uint32_t numberOfIndices,
    uint32_t const * indices)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(requestConfigId);
        device.AttachActiveList(requestConfigId, operationIndex, numberOfIndices, indices);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigEnableHardwareConsistency(
    uint32_t requestConfigId,
    enum Gna2DeviceVersion deviceVersion)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(requestConfigId);
        device.EnableHardwareConsistency(requestConfigId, deviceVersion);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigSetAccelerationMode(
    uint32_t requestConfigId,
    enum Gna2AccelerationMode accelerationMode)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(requestConfigId);
        device.EnforceAcceleration(requestConfigId, accelerationMode);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigRelease(
    uint32_t requestConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(requestConfigId);
        device.ReleaseConfiguration(requestConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestEnqueue(
    uint32_t requestConfigId,
    uint32_t * requestId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(requestConfigId);
        device.PropagateRequest(requestConfigId, requestId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestWait(
    uint32_t requestId,
    uint32_t timeoutMilliseconds)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestId(requestId);
        return device.WaitForRequest(requestId, timeoutMilliseconds);
    };
    return ApiWrapper::ExecuteSafely(command);
}

AccelerationMode::AccelerationMode(Gna2AccelerationMode basicMode, bool hardwareConsistencyEnabled)
    : hardwareConsistency{ hardwareConsistencyEnabled }
{
    SetMode(basicMode);
    enforceValidity();
}

bool AccelerationMode::IsHardwareEnforced() const
{
    return mode == Gna2AccelerationModeHardware;
}

bool AccelerationMode::IsSoftwareEnforced() const
{
    return mode == Gna2AccelerationModeSoftware ||
        mode == Gna2AccelerationModeGeneric ||
        mode == Gna2AccelerationModeSse4x2 ||
        mode == Gna2AccelerationModeAvx1 ||
        mode == Gna2AccelerationModeAvx2;
}

AccelerationMode AccelerationMode::GetEffectiveSoftwareAccelerationMode(
    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerations) const
{
    if (mode == Gna2AccelerationModeHardware)
    {
        throw GnaException(Gna2StatusAccelerationModeNotSupported);
    }
    if (mode == Gna2AccelerationModeSoftware ||
        mode == Gna2AccelerationModeAuto)
    {
        //last is fastest
        return AccelerationMode{ supportedCpuAccelerations.back(), hardwareConsistency };
    }
    for(const auto& supported: supportedCpuAccelerations)
    {
        if(mode == supported)
        {
            return AccelerationMode{ supported, hardwareConsistency };
        }
    }
    throw GnaException(Gna2StatusAccelerationModeNotSupported);
}

void AccelerationMode::ExpectValid(Gna2AccelerationMode modeIn)
{
    static const std::vector<Gna2AccelerationMode> existingModes = {
        Gna2AccelerationModeAuto,
        Gna2AccelerationModeSoftware,
        Gna2AccelerationModeHardware,
        Gna2AccelerationModeAvx2,
        Gna2AccelerationModeAvx1,
        Gna2AccelerationModeSse4x2,
        Gna2AccelerationModeGeneric,
    };
    Expect::InSet(modeIn, existingModes, Gna2StatusAccelerationModeNotSupported);
}

void AccelerationMode::SetMode(Gna2AccelerationMode newMode)
{
    ExpectValid(newMode);
    mode = newMode;
    enforceValidity();
}

static std::map<AccelerationMode, const char*> AccelerationModeNames{
    {AccelerationMode{ Gna2AccelerationModeHardware },"GNA_HW"},
    {AccelerationMode{ Gna2AccelerationModeAuto,true },"GNA_AUTO_SAT"},
    {AccelerationMode{ Gna2AccelerationModeAuto,false },"GNA_AUTO_FAST"},
    {AccelerationMode{ Gna2AccelerationModeSoftware,true },"GNA_SW_SAT"},
    {AccelerationMode{ Gna2AccelerationModeSoftware,false },"GNA_SW_FAST"},
    {AccelerationMode{ Gna2AccelerationModeGeneric,true },"GNA_GEN_SAT"},
    {AccelerationMode{ Gna2AccelerationModeGeneric,false },"GNA_GEN_FAST"},
    {AccelerationMode{ Gna2AccelerationModeSse4x2,true },"GNA_SSE4_2_SAT"},
    {AccelerationMode{ Gna2AccelerationModeSse4x2,false },"GNA_SSE4_2_FAST"},
    {AccelerationMode{ Gna2AccelerationModeAvx1, true },"GNA_AVX1_SAT"},
    {AccelerationMode{ Gna2AccelerationModeAvx1, false },"GNA_AVX1_FAST"},
    {AccelerationMode{ Gna2AccelerationModeAvx2,true },"GNA_AVX2_SAT"},
    {AccelerationMode{ Gna2AccelerationModeAvx2,false },"GNA_AVX2_FAST"},
};

const char* AccelerationMode::UNKNOWN_ACCELERATION_MODE_NAME = "GNA_UNKNOWN_ACCELERATION_MODE";

const char* AccelerationMode::GetName() const
{
    auto item = AccelerationModeNames.find(*this);
    if (item != AccelerationModeNames.end())
    {
        return item->second;
    }
    return UNKNOWN_ACCELERATION_MODE_NAME;
}

void AccelerationMode::SetHwConsistency(bool consistencyEnabled)
{
    hardwareConsistency = consistencyEnabled;
    enforceValidity();
}

bool AccelerationMode::GetHwConsistency() const
{
    return hardwareConsistency;
}

Gna2AccelerationMode GNA::AccelerationMode::GetMode() const
{
    return mode;
}

void AccelerationMode::enforceValidity()
{
    if (mode == Gna2AccelerationModeHardware)
    {
        hardwareConsistency = true;
    }
}

bool AccelerationMode::operator<(const AccelerationMode& right) const
{
    const auto ret = (mode < right.mode) || ((mode == right.mode) && hardwareConsistency && (!right.hardwareConsistency));
    return ret;
}
