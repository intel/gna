/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "HardwareModel.h"

#include "Address.h"
#include "HardwareCapabilities.h"
#include "ModelExportConfig.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{
class LayerDescriptor;

class HardwareModelNoMMU : public HardwareModel
{
public:

    HardwareModelNoMMU(CompiledModel const & softwareModel, Gna2UserAllocator customAlloc, Gna2DeviceVersion targetDevice);

    virtual ~HardwareModelNoMMU() = default;

    const LayerDescriptor& GetDescriptor(uint32_t layerIndex) const;

    uint32_t GetOutputOffset(uint32_t layerIndex) const;

    uint32_t GetInputOffset(uint32_t layerIndex) const;

    static uint32_t SetBarIndex(uint32_t offsetFromBar, uint32_t barIndex);

    // Adds BAR index at low 2 bits
    virtual LdaOffset GetBufferOffset(const BaseAddress& address) const override;

    void ExportComponent(void *& exportData, uint32_t & exportDataSize, Gna2ModelExportComponent component);

    static constexpr uint32_t GnaDescriptorSize = 32;

protected:
    virtual void prepareAllocationsAndModel() override;

    static uint32_t GetBarIndex(Gna2DeviceVersion target, uint32_t tag);

    static std::map<uint32_t, uint32_t> const & GetBarMap(Gna2DeviceVersion target);

    MemoryContainer const & GetComponent(Gna2ModelExportComponent component) const;

    static Gna2MemoryTag GetComponentTag(Gna2DeviceVersion target,
        Gna2ModelExportComponent component);

    void PrepareExportAllocations();
    void GuessIOAllocations();

    std::unique_ptr<Memory> allocLD(uint32_t ldMemorySize, uint32_t ldSize = Memory::GNA_BUFFER_ALIGNMENT) override;

private:
    void ExportLd(void *& exportData, uint32_t & exportDataSize);
    static const HardwareCapabilities& GetHwCaps(Gna2DeviceVersion targetDevice);

    Gna2UserAllocator customUserAlloc = nullptr;

    void* customAllocSafe(uint32_t size)
    {
        Expect::NotNull(customUserAlloc);
        auto o = customUserAlloc(size);
        Expect::NotNull(o, Gna2StatusResourceAllocationError);
        return o;
    }

    std::map<Gna2MemoryTag, MemoryContainer> exportAllocations;

    std::unique_ptr<Memory> guessedInput;
    std::unique_ptr<Memory> guessedOutput;
};

}
