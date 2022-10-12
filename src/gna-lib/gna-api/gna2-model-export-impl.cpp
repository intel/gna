/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "gna2-model-export-impl.h"

#include "ApiWrapper.h"
#include "Device.h"
#include "ModelExportConfig.h"

#include "gna2-common-impl.h"
using namespace GNA;

GNA2_API enum Gna2Status Gna2ModelExportConfigCreate(
    Gna2UserAllocator userAllocator,
    uint32_t * const exportConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(userAllocator);
        Expect::NotNull(exportConfigId);
        *exportConfigId = ModelExportManager::Get().AddConfig(userAllocator);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelExportConfigRelease(
    uint32_t exportConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        ModelExportManager::Get().RemoveConfig(exportConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelExportConfigSetSource(
    uint32_t exportConfigId,
    uint32_t sourceDeviceIndex,
    uint32_t sourceModelId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& config = ModelExportManager::Get().GetConfig(exportConfigId);
        config.SetSource(sourceDeviceIndex, sourceModelId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelExportConfigSetTarget(
    uint32_t exportConfigId,
    enum Gna2DeviceVersion targetDeviceVersion)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& config = ModelExportManager::Get().GetConfig(exportConfigId);
        config.SetTarget(targetDeviceVersion);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelExport(
    uint32_t exportConfigId,
    enum Gna2ModelExportComponent componentType,
    void ** exportBuffer,
    uint32_t * exportBufferSize)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& config = ModelExportManager::Get().GetConfig(exportConfigId);
        config.Export(componentType, exportBuffer, exportBufferSize);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelOverrideAlignment(
    uint32_t newAlignment)
{
    const std::function<ApiStatus()> command = [&]()
    {
        TensorLimits::OverrideAlign(newAlignment);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}
