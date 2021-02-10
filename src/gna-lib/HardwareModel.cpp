/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "HardwareModel.h"

#include "ActivationFunction.h"
#include "common.h"
#include "CompiledModel.h"
#include "Expect.h"
#include "GnaConfig.h"
#include "GnaException.h"
#include "HardwareCapabilities.h"
#include "HardwareLayer.h"
#include "Layer.h"
#include "Memory.h"
#include "SubModel.h"
#include "TransformMap.h"

#include "gna-api-status.h"
#include "gna-api-types-xnn.h"

#include <algorithm>

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
    HwModule{ HwModuleInterface::Create("gna_hw") }
{
}

bool HardwareModel::IsSoftwareLayer(const std::vector<std::unique_ptr<SubModel>>& submodels, uint32_t layerIndex)
{
    for (const auto& subModel : submodels)
    {
        if (layerIndex >= subModel->LayerIndex && layerIndex < subModel->LayerIndex + subModel->GetLayerCount() &&
            subModel->Type == SubmodelType::Software)
        {
            return true;
        }
    }
    return false;
}

void HardwareModel::Build(const std::vector<std::unique_ptr<SubModel>>& submodels)
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
    auto i = uint32_t { 0 };
    for (auto const & layerIter : model.GetLayers())
    {
        try
        {
            auto const & layer = *layerIter;
            const auto parameters = DescriptorParameters{layer, layerDescriptor, *HwModule };
            if (IsSoftwareLayer(submodels, i))
            {
                hardwareLayers.push_back(nullptr);
            }
            else
            {
                hardwareLayers.push_back(HardwareLayer::Create(parameters));
            }
            if (INTEL_GMM == layer.Operation)
            {
                gmmDescriptor++;
            }
            layerDescriptor.Forward(gmmDescriptor);
            i++;
        }
        catch (GnaModelErrorException& e)
        {
            e.SetLayerIndex(i);
            throw;
        }
        catch (const GnaException& e)
        {
            if (e.GetStatus() == Gna2StatusHardwareModuleNotFound ||
                e.GetStatus() == Gna2StatusHardwareModuleSymbolNotFound)
            {
                throw;
            }
            throw GnaModelErrorException(i, e.GetStatus());
        }
        catch (...)
        {
            throw GnaModelErrorException(i);
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

uint32_t HardwareModel::GetBufferOffset(const BaseAddress& address) const
{
    return allocations.GetBufferOffset(address, PAGE_SIZE);
}

uint32_t HardwareModel::getLayerDescriptorsSize(
    const uint32_t layerCount, const DeviceVersion deviceVersion)
{
    auto layerDescriptorsSizeTmp = LayerDescriptor::GetSize(layerCount, deviceVersion);
    return layerDescriptorsSizeTmp;
}

uint32_t HardwareModel::getGmmDescriptorsSize(const uint32_t gmmLayersCount)
{
    auto const gmmDescriptorsSizeTmp = size_t{gmmLayersCount * sizeof(GMM_CONFIG)};
    return static_cast<uint32_t>(gmmDescriptorsSizeTmp);
}

void HardwareModel::prepareAllocationsAndModel()
{
    Expect::InRange(model.LayerCount, ui32_1, HardwareCapabilities::GetMaximumLayerCount(DefaultDeviceVersion),
        Gna2StatusXnnErrorNetLyrNo);
    auto ldMemorySize = calculateDescriptorSize(true);
    auto ldSize = LayerDescriptor::GetSize(1, hwCapabilities.GetDeviceVersion());

    ldMemory = std::make_unique<Memory>(ldMemorySize, ldSize);
    if (!ldMemory)
    {
        throw GnaException {Gna2StatusResourceAllocationError};
    }

    prepareBaseDescriptor();

    allocations.Append(model.GetAllocations());

    auto const modelSize = allocations.GetMemorySizeAlignedToPage();
    Expect::InRange(modelSize, hwCapabilities.MaximumModelSize,
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
