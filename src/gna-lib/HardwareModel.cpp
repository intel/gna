/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "HardwareModel.h"

#include "CompiledModel.h"
#include "Expect.h"
#include "GnaConfig.h"
#include "GnaException.h"
#include "HardwareCapabilities.h"
#include "HardwareLayer.h"
#include "Layer.h"
#include "Memory.h"
#include "TransformMap.h"

#include "gna2-model-export-api.h"

#include <sstream>
#include <cinttypes>
#include <iomanip>

using namespace GNA;

uint32_t HardwareModel::calculateDescriptorSize(bool includeGmms) const
{
    auto const gmmDescriptorsSizeTmp = includeGmms ? gmmDescriptorsSize : 0;

    return xnnDescriptorsSize + gmmDescriptorsSizeTmp;
}

HardwareModel::HardwareModel(CompiledModel const & softwareModel, const HardwareCapabilities& hwCaps) :
    model{ softwareModel },
    hwCapabilities{ hwCaps },
    gmmDescriptorsSize{ getGmmDescriptorsSize(model.GmmCount) },
    xnnDescriptorsSize{ getLayerDescriptorsSize(model.LayerCount, hwCapabilities.GetDeviceVersion()) },
    HwModule{ hwCapabilities.GetDeviceVersion() }
{
}

void HardwareModel::Build()
{
    prepareAllocationsAndModel();

    auto gmmDescriptor = AddrGmmCfg();
    if (0 != gmmDescriptorsSize)
    {
        gmmDescriptor = AddrGmmCfg(ldMemory->GetBuffer<uint8_t>() +
            LayerDescriptor::GetSize(model.LayerCount, hwCapabilities.GetDeviceVersion()));
    }
    auto layerDescriptor = LayerDescriptor(*baseDescriptor, gmmDescriptor,
        getHwOffsetFunction);
    auto i = uint32_t{ 0 };
    for (auto const & layerIter : model.GetLayers())
    {
        try
        {
            auto const & layer = *layerIter;
            const auto parameters = DescriptorParameters{ layer, layerDescriptor, HwModule };
            if (IsSoftwareLayer(i))
            {
                hardwareLayers.push_back(nullptr);
            }
            else
            {
                hardwareLayers.push_back(HardwareLayer::Create(parameters));
#if DEBUG == 1
                std::stringstream descriptorStream;
                Log->Message("Layer %d descriptor :\n", i);
                descriptorStream << "\n";
                const auto addr = hardwareLayers.back()->XnnDescriptor.GetMemAddress().Get();
                for (unsigned line = 0; line < 8; line++)
                {
                    descriptorStream << std::hex << std::setw(11) << reinterpret_cast<uint64_t>(addr + line * 8) << " ";
                    for (auto byte = 0u; byte < 8; byte++)
                    {
                        descriptorStream << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*(addr + line * 8 + byte)) << " ";
                    }
                    descriptorStream << "\n";
                }
                Log->Message(descriptorStream.str().c_str());
#endif // DEBUG == 1
            }
            if (INTEL_GMM == layer.Operation)
            {
                gmmDescriptor++;
            }
            layerDescriptor.Forward(gmmDescriptor);
            i++;
        }
        catch (const GnaException& e)
        {
            if (e.GetStatus() == Gna2StatusHardwareModuleNotFound ||
                e.GetStatus() == Gna2StatusHardwareModuleSymbolNotFound)
            {
                throw;
            }
            GnaModelErrorException::DispatchAndSetLayer(i);
        }
        catch (...)
        {
            GnaModelErrorException::DispatchAndSetLayer(i);
        }
    }
}

HardwareLayer const & HardwareModel::GetLayer(uint32_t layerIndex) const
{
    auto const layer = TryGetLayer(layerIndex);
    if (nullptr != layer)
    {
        return *layer;
    }
    throw GnaException(Gna2StatusXnnErrorLyrCfg);
}

HardwareLayer const * HardwareModel::TryGetLayer(uint32_t layerIndex) const
{
    try
    {
        return hardwareLayers.at(layerIndex).get();
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

LdaOffset HardwareModel::GetBufferOffset(const BaseAddress& address) const
{
    return allocations.GetBufferOffset(address, MemoryBufferAlignment);
}

uint32_t HardwareModel::getLayerDescriptorsSize(
    uint32_t layerCount, DeviceVersion deviceVersion)
{
    const auto layerDescriptorsSizeTmp = LayerDescriptor::GetSize(layerCount, deviceVersion);
    return layerDescriptorsSizeTmp;
}

uint32_t HardwareModel::getGmmDescriptorsSize(uint32_t gmmLayersCount)
{
    auto const gmmDescriptorsSizeTmp = size_t{ gmmLayersCount * sizeof(GMM_CONFIG) };
    return static_cast<uint32_t>(gmmDescriptorsSizeTmp);
}

void HardwareModel::prepareAllocationsAndModel()
{
    auto ldMemorySize = calculateDescriptorSize(true);
    auto ldSize = LayerDescriptor::GetSize(1, hwCapabilities.GetDeviceVersion());

    ldMemory = allocLD(ldMemorySize, ldSize);

    if (!ldMemory)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }

    prepareBaseDescriptor();

    allocations.Append(model.GetAllocations());


    auto const modelSize = allocations.GetMemorySizeAlignedToPage();
    Expect::InRange(modelSize, HardwareCapabilities::MaximumModelSize,
        Gna2StatusMemoryTotalSizeExceeded);

    getHwOffsetFunction = [this](const BaseAddress& buffer) { return GetBufferOffset(buffer); };
}

void HardwareModel::prepareBaseDescriptor()
{
    baseDescriptor = std::make_unique<LayerDescriptor>(
        *ldMemory, ldMemory->GetBuffer(), hwCapabilities);
    if (!baseDescriptor)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }

    // make ensure it's first on a list
    allocations.Emplace(*ldMemory);
}

void HardwareModel::createScratchPadMemory(void * buffer, uint32_t size)
{
    scratchPadMemory = std::make_unique<Memory>(buffer, size);
    if (!scratchPadMemory)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }
    scratchPadMemory->SetTag(Gna2MemoryTagScratch);
}

bool HardwareModel::IsSoftwareLayer(uint32_t layerIndex) const
{
    UNREFERENCED_PARAMETER(layerIndex);
    return false;
}

std::unique_ptr<Memory> HardwareModel::allocLD(uint32_t ldMemorySize, uint32_t ldSize)
{
    return std::make_unique<Memory>(ldMemorySize, ldSize);
}

HardwareModelTarget::HardwareModelTarget(CompiledModel const& softwareModel, const HardwareCapabilities& hwCaps) :
    HardwareModel(softwareModel, hwCaps)
{
    Build();
}
