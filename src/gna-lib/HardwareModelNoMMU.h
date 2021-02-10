/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "HardwareModel.h"

#include "Address.h"
#include "HardwareCapabilities.h"


#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{
class LayerDescriptor;

class HardwareModelNoMMU : public HardwareModel
{
public:

    HardwareModelNoMMU(CompiledModel const & softwareModel, Gna2UserAllocator customAlloc);

    virtual ~HardwareModelNoMMU() = default;

    const LayerDescriptor& GetDescriptor(uint32_t layerIndex) const;

    uint32_t GetOutputOffset(uint32_t layerIndex) const;

    uint32_t GetInputOffset(uint32_t layerIndex) const;

    static uint32_t SetBarIndex(uint32_t offsetFromBar, uint32_t barIndex);

    // Adds BAR index at low 2 bits
    virtual uint32_t GetBufferOffset(const BaseAddress& address) const override;

    void ExportLd(void *& exportData, uint32_t & exportDataSize);

    void * ROBeginAddress = nullptr;
    void * InputAddress = nullptr;
    void * OutputAddress = nullptr;

    static constexpr uint32_t GnaDescritorSize = 32;
    static constexpr uint32_t BarIndexGnaBar = 0;
    static constexpr uint32_t BarIndexRo = 1;
    static constexpr uint32_t BarIndexInput = 2;
    static constexpr uint32_t BarIndexOutput = 3;

protected:
    virtual void prepareAllocationsAndModel() override;

private:
    static HardwareCapabilities noMMUCapabilities;

    Gna2UserAllocator customAlloc = nullptr;
};

}
