/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "HardwareModelNoMMU.h"

#include "CompiledModel.h"
#include "Memory.h"
#include <AffineLayers.h>

using namespace GNA;

HardwareCapabilities HardwareModelNoMMU::noMMUCapabilities = HardwareCapabilities{ Gna2DeviceVersionFromInt(0x30E) };

HardwareModelNoMMU::HardwareModelNoMMU(CompiledModel const & softwareModel, Gna2UserAllocator customAllocIn) :
    HardwareModel(softwareModel, noMMUCapabilities),
    customAlloc{ customAllocIn }
{
    InputAddress = softwareModel.GetLayers().front()->Input.Buffer.Get();
    OutputAddress = softwareModel.GetLayers().back()->Output.Buffer.Get();
}

const LayerDescriptor& HardwareModelNoMMU::GetDescriptor(uint32_t layerIndex) const
{
    return GetLayer(layerIndex).XnnDescriptor;
}

uint32_t HardwareModelNoMMU::GetOutputOffset(uint32_t layerIndex) const
{
    auto layer = GetLayer(layerIndex);
    return layer.GetLdOutputOffset() - GetDescriptor(0).GetOffset();
}

uint32_t HardwareModelNoMMU::GetInputOffset(uint32_t layerIndex) const
{
    auto layer = GetLayer(layerIndex);
    return layer.GetLdInputOffset() - GetDescriptor(0).GetOffset();
}

void HardwareModelNoMMU::prepareAllocationsAndModel()
{
    Expect::InRange(model.LayerCount, ui32_1, hwCapabilities.GetMaximumLayerCount(),
        Gna2StatusXnnErrorNetLyrNo);
    auto const ldMemorySize = HardwareModel::calculateDescriptorSize(false);

    ldMemory = std::make_unique<Memory>(customAlloc(ldMemorySize), ldMemorySize);

    if (ldMemory->GetBuffer() == nullptr)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }
    memset(ldMemory->GetBuffer(), 0, ldMemorySize);

    prepareBaseDescriptor();

    getHwOffsetFunction = [this](const BaseAddress& buffer) { return GetBufferOffset(buffer); };
}

uint32_t HardwareModelNoMMU::SetBarIndex(uint32_t offsetFromBar, uint32_t barIndex)
{
    return offsetFromBar | barIndex;
}

uint32_t HardwareModelNoMMU::GetBufferOffset(const BaseAddress& address) const
{
    if (address == AffineBaseLayer::GetGlobal2MBScratchpad())
    {
        // Global scratchpad region starts after GnaDescriptor (32bytes) at BAR0
        return SetBarIndex(GnaDescritorSize, BarIndexGnaBar);
    }
    if (address == InputAddress)
    {
        return  SetBarIndex(0, BarIndexInput);
    }
    if (address == OutputAddress)
    {
        return SetBarIndex(0, BarIndexOutput);
    }
    return SetBarIndex(ldMemory->GetSize() + address.GetOffset( BaseAddress(ROBeginAddress)), BarIndexRo);
}

void HardwareModelNoMMU::ExportLd(void *& exportData, uint32_t & exportDataSize)
{
    Expect::NotNull(ROBeginAddress);
    Build({});

    exportData = ldMemory->GetBuffer();
    exportDataSize = ldMemory->GetSize();
}
