/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Address.h"
#include "HardwareLayer.h"
#include "HwModuleInterface.hpp"
#include "LayerDescriptor.h"
#include "MemoryContainer.h"
#include "Memory.h"

#include <cstdint>
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
    virtual ~HardwareModel() = default;

    HardwareModel(const HardwareModel&) = delete;
    HardwareModel(HardwareModel&&) = delete;
    HardwareModel& operator=(const HardwareModel&) = delete;
    HardwareModel& operator=(HardwareModel&&) = delete;

    HardwareLayer const & GetLayer(uint32_t layerIndex) const;

    HardwareLayer const * TryGetLayer(uint32_t layerIndex) const;

    /* Calculates offset proper for GNA hardware
     * Few assumptions here:
     * a) MMU is enabled
     * b) layer descriptor memory is added first to MMU
     * c) other memory buffers are added to MMU in order they are provided
     */
    virtual LdaOffset GetBufferOffset(const BaseAddress& address) const;

protected:
    HardwareModel(CompiledModel const & softwareModel, const HardwareCapabilities& hwCaps);

    uint32_t calculateDescriptorSize(bool includeGmms) const;

    static uint32_t getLayerDescriptorsSize(uint32_t layerCount,
        DeviceVersion deviceVersion = DefaultDeviceVersion);
    static uint32_t getGmmDescriptorsSize(uint32_t gmmLayersCount);

    virtual void prepareAllocationsAndModel();

    void prepareBaseDescriptor();

    void createScratchPadMemory(void * buffer, uint32_t size);

    virtual bool IsSoftwareLayer(uint32_t layerIndex) const;

    void Build();

    std::unique_ptr<LayerDescriptor> baseDescriptor;

    CompiledModel const & model;

    const HardwareCapabilities& hwCapabilities;

    const uint32_t gmmDescriptorsSize;

    const uint32_t xnnDescriptorsSize;

    std::vector<std::unique_ptr<HardwareLayer>> hardwareLayers;

    std::unique_ptr<Memory> ldMemory;

    std::unique_ptr<Memory> scratchPadMemory;

    // hardware model (ldMemory) + software model allocations
    MemoryContainer allocations;

    const HwModuleInterface HwModule;

    GetHwOffset getHwOffsetFunction;
};

// no export available, only for purpose of building target HW model for verification
class HardwareModelTarget final : public HardwareModel
{
public:
    HardwareModelTarget(CompiledModel const & softwareModel, const HardwareCapabilities& hwCaps);

    HardwareModelTarget(const HardwareModelTarget&) = delete;
    HardwareModelTarget(HardwareModelTarget&&) = delete;
    HardwareModelTarget& operator=(const HardwareModelTarget&) = delete;
    HardwareModelTarget& operator=(HardwareModelTarget&&) = delete;

    virtual ~HardwareModelTarget() = default;
};

}

