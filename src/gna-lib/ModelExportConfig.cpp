/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "ModelExportConfig.h"

#include "DeviceManager.h"
#include "GnaException.h"

#include "gna2-model-export-impl.h"
#include "gna2-model-suecreek-header.h"

#include <map>
#include <cstdint>

using namespace GNA;

ModelExportConfig::ModelExportConfig(Gna2UserAllocator userAllocatorIn) :
    userAllocator{ userAllocatorIn }
{
    Expect::NotNull((void *)userAllocator);
}

void ModelExportConfig::Export(Gna2ModelExportComponent componentType, void ** exportBuffer, uint32_t * exportBufferSize)
{
    Expect::NotNull(exportBufferSize);
    Expect::NotNull(exportBuffer);

    ValidateState();

    *exportBuffer = nullptr;
    *exportBufferSize = 0;
    Gna2Status status;
    auto& device = DeviceManager::Get().GetDevice(sourceDeviceId);
    if (componentType == Gna2ModelExportComponentLegacySueCreekHeader)
    {
        *exportBufferSize = sizeof(Gna2ModelSueCreekHeader);
        const auto header = static_cast<Gna2ModelSueCreekHeader *>(userAllocator(*exportBufferSize));
        *exportBuffer = header;
        const auto dump = device.Dump(sourceModelId,
            header, &status, privateAllocator);
        privateDeAllocator(dump);
        return;
    }

    if (componentType == Gna2ModelExportComponentLegacySueCreekDump)
    {
        Gna2ModelSueCreekHeader header = {};
        *exportBuffer = device.Dump(sourceModelId, &header, &status, userAllocator);
        *exportBufferSize = header.ModelSize;
        return;
    }

    if (componentType == Gna2ModelExportComponentLayerDescriptors)
    {
        device.DumpLdNoMMu(sourceModelId, userAllocator, *exportBuffer, *exportBufferSize);
        return;
    }

    throw GnaException(Gna2StatusNotImplemented);
}

void ModelExportConfig::SetSource(uint32_t deviceId, uint32_t modelId)
{
    sourceDeviceId = deviceId;
    sourceModelId = modelId;
}

void ModelExportConfig::SetTarget(Gna2DeviceVersion version)
{
    targetDeviceVersion = version;
}

void ModelExportConfig::ValidateState() const
{
    Expect::NotNull((void *)userAllocator);
    Expect::True(sourceDeviceId != Gna2DisabledU32, Gna2StatusIdentifierInvalid);
    Expect::True(sourceModelId != Gna2DisabledU32, Gna2StatusIdentifierInvalid);
    uint32_t const legacySueCreekVersionNumber = 0xFFFF0001;
    auto const is1x0Embedded = Gna2DeviceVersionFromInt(0x10E) == targetDeviceVersion
    || legacySueCreekVersionNumber == static_cast<uint32_t>(targetDeviceVersion);
    auto const is3x0Embedded = targetDeviceVersion == Gna2DeviceVersionFromInt(0x30E);
    Expect::True(is1x0Embedded || is3x0Embedded, Gna2StatusAccelerationModeNotSupported);
}

inline void * ModelExportConfig::privateAllocator(uint32_t size)
{
    return _mm_malloc(size, 4096);
}

inline void ModelExportConfig::privateDeAllocator(void * ptr)
{
    _mm_free(ptr);
}

ModelExportManager & ModelExportManager::Get()
{
    static ModelExportManager globalManager;
    return globalManager;
}

uint32_t ModelExportManager::AddConfig(Gna2UserAllocator userAllocator)
{
    auto idCreated = configCount++;
    allConfigs.emplace(idCreated, ModelExportConfig{ userAllocator });
    return idCreated;
}

void ModelExportManager::RemoveConfig(uint32_t id)
{
    allConfigs.erase(id);
}

ModelExportConfig & ModelExportManager::GetConfig(uint32_t exportConfigId)
{
    auto found = allConfigs.find(exportConfigId);
    if (found == allConfigs.end())
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
    return found->second;
}
