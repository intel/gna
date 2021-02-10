/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Address.h"
#include "HardwareLayer.h"
#include "HwModuleInterface.hpp"
#include "KernelArguments.h"
#include "LayerDescriptor.h"
#include "MemoryContainer.h"
#include "SubModel.h"

#include "gna-api.h"

#include <cstdint>
#include <map>
#include <memory>

#include "gna2-common-impl.h"

namespace GNA
{

class CompiledModel;

class HardwareCapabilities;

class Layer;

class HardwareModel
{
public:
    HardwareModel(CompiledModel const & softwareModel, const HardwareCapabilities& hwCaps);

    HardwareModel(const HardwareModel &) = delete;
    HardwareModel& operator=(const HardwareModel&) = delete;
    virtual ~HardwareModel() = default;

    void Build(const std::vector<std::unique_ptr<SubModel>>& submodels);

    HardwareLayer const & GetLayer(uint32_t layerIndex) const;

    HardwareLayer const * TryGetLayer(uint32_t layerIndex) const;

    /* Calculates offset proper for GNA hardware
     * Few assumptions here:
     * a) MMU is enabled
     * b) layer descriptor memory is added first to MMU
     * c) other memory buffers are added to MMU in order they are provided
     */
    virtual uint32_t GetBufferOffset(const BaseAddress& address) const;

protected:
    uint32_t calculateDescriptorSize(bool includeGmms) const;

    static uint32_t getLayerDescriptorsSize(const uint32_t layerCount,
        DeviceVersion deviceVersion = DefaultDeviceVersion);
    static uint32_t getGmmDescriptorsSize(const uint32_t gmmLayersCount);

    virtual void prepareAllocationsAndModel();

    void prepareBaseDescriptor();

    bool IsSoftwareLayer(const std::vector<std::unique_ptr<SubModel>>& submodels, uint32_t layerIndex);

    std::unique_ptr<LayerDescriptor> baseDescriptor;

    CompiledModel const & model;

    const HardwareCapabilities& hwCapabilities;

    const uint32_t gmmDescriptorsSize;

    const uint32_t xnnDescriptorsSize;

    std::vector<std::unique_ptr<HardwareLayer>> hardwareLayers;

    std::unique_ptr<Memory> ldMemory;

    // hardware model (ldMemory) + software model allocations
    MemoryContainer allocations;

    std::unique_ptr<HwModuleInterface const> const HwModule;

    GetHwOffset getHwOffsetFunction;
};

}
