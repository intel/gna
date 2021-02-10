/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "gna2-model-export-api.h"

#include "gna2-common-api.h"
#include "gna2-common-impl.h"

#include <map>

namespace GNA
{
class ModelExportConfig
{
public:
    explicit ModelExportConfig(Gna2UserAllocator userAllocatorIn);
    void SetSource(uint32_t deviceId, uint32_t modelId);
    void SetTarget(Gna2DeviceVersion version);
    void Export(enum Gna2ModelExportComponent componentType,
        void ** exportBuffer,
        uint32_t * exportBufferSize);

protected:
    void ValidateState() const;

private:
    Gna2UserAllocator userAllocator = nullptr;
    uint32_t sourceDeviceId = Gna2DisabledU32;
    uint32_t sourceModelId = Gna2DisabledU32;
    Gna2DeviceVersion targetDeviceVersion = Gna2DeviceVersionSoftwareEmulation;

    static void* privateAllocator(uint32_t size);
    static void privateDeAllocator(void * ptr);
};

class ModelExportManager
{
public:
    static ModelExportManager& Get();

    ModelExportManager(const ModelExportManager&) = delete;
    void operator=(const ModelExportManager&) = delete;

    uint32_t AddConfig(Gna2UserAllocator userAllocator);
    void RemoveConfig(uint32_t id);
    ModelExportConfig& GetConfig(uint32_t exportConfigId);

private:
    ModelExportManager() = default;
    uint32_t configCount = 0;
    std::map<uint32_t, ModelExportConfig> allConfigs;
};

}
