/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "HardwareRequest.h"

#include "ActiveList.h"
#include "Address.h"
#include "CompiledModel.h"
#include "GnaException.h"
#include "HardwareLayer.h"
#include "HardwareModelScorable.h"
#include "Layer.h"
#include "LayerConfiguration.h"
#include "RecurrentFunction.h"
#include "RequestConfiguration.h"


#include <utility>

using namespace GNA;

HardwareRequest::HardwareRequest(const HardwareModelScorable& hwModelIn,
    const RequestConfiguration& requestConfigurationIn,
    MemoryContainer const & modelAllocations) :
    HwPerfEncoding(requestConfigurationIn.GetHwInstrumentationMode()),
    RequestConfigId(requestConfigurationIn.Id),
    requestConfiguration(requestConfigurationIn),
    hwModel(hwModelIn)
{
    modelAllocations.CopyEntriesTo(DriverMemoryObjects);
    requestConfiguration.GetAllocations().CopyEntriesTo(DriverMemoryObjects);
    Invalidate();
}

void HardwareRequest::Invalidate()
{
    auto& ldPatches = DriverMemoryObjects.front().Patches;
    ldPatches.clear();

    auto& model = requestConfiguration.Model;
    auto& layerConfigurations = requestConfiguration.LayerConfigurations;

    for (auto it = layerConfigurations.cbegin(); it != layerConfigurations.cend(); ++it)
    {
        auto const & layer = model.GetLayer(it->first);
        auto const hwLayer = hwModel.TryGetLayer(it->first);
        if (hwLayer == nullptr)
        {
            continue;
        }
        auto const layerCfg = it->second.get();

        generateBufferPatches(*layerCfg, layer, *hwLayer);

        if (layer.Operation == INTEL_AFFINE || layer.Operation == INTEL_GMM)
        {
            auto const nnopTypeOffset = hwLayer->GetLdNnopOffset();
            auto const nnopTypeValue = hwLayer->GetNnopType(layerCfg->ActList != nullptr);
            ldPatches.push_back({ nnopTypeOffset, nnopTypeValue, sizeof(uint8_t) });
        }

        if (layerCfg->ActList)
        {
            const auto activeList = it->second->ActList.get();

            const auto ldActlistOffset = hwLayer->GetLdActlistOffset();
            const auto actlistOffset = hwModel.GetBufferOffsetForConfiguration(
                activeList->Indices, requestConfiguration);
            const auto ldActlenOffset = hwLayer->GetLdActlenOffset();

            ldPatches.push_back({ ldActlistOffset, actlistOffset, sizeof(ASLADDR) });
            ldPatches.push_back({ ldActlenOffset, activeList->IndicesCount, sizeof(ASTLISTLEN) });

            if (INTEL_GMM == layer.Operation)
            {
                const auto ldScrlenOffset = hwLayer->GetLdScrlenOffset();
                const auto scrlen = hwLayer->GetScrlen(activeList->IndicesCount);

                ldPatches.push_back({ ldScrlenOffset, scrlen, sizeof(GMMSCRLEN) });
            }
        }
    }
}

void HardwareRequest::Update(uint32_t layerIndex, uint32_t layerCount, GnaOperationMode mode)
{
    auto const & hwLayer = hwModel.GetLayer(layerIndex);

    Mode = mode;
    LayerCount = layerCount;
    LayerBase = hwLayer.GetXnnDescriptorOffset();
    if (GMM == mode)
    {
        GmmOffset = hwLayer.GetGmmDescriptorOffset();
        updateGmmModeActiveLists(layerIndex, layerCount);
        GmmModeActiveListOn = gmmModeActiveLists.at(layerIndex);
    }
}

void HardwareRequest::generateBufferPatches(const LayerConfiguration& layerConfiguration,
    const Layer &layer, const HardwareLayer &hwLayer)
{
    const auto& buffers = layerConfiguration.Buffers;
    auto& ldPatches = DriverMemoryObjects.front().Patches;

    for (auto it = buffers.cbegin(); it != buffers.cend(); it++)
    {
        auto componentType = it->first;
        auto address = it->second;

        uint32_t bufferOffset = hwModel.GetBufferOffsetForConfiguration(address, requestConfiguration);
        uint32_t ldOffset = 0;
        switch (componentType)
        {
        case InputOperandIndex:
            ldOffset = hwLayer.GetLdInputOffset();
            break;
        case OutputOperandIndex:
        {
            ldOffset = hwLayer.GetLdOutputOffset();
            if (INTEL_RECURRENT == layer.Operation)
            {
                auto const & recurrentFunction = layer.Transforms.Get<RecurrentFunction>(RecurrentTransform);
                auto const newFbAddress = recurrentFunction.CalculateFeedbackBuffer(address);
                auto const feedbackBufferOffset = hwModel.GetBufferOffsetForConfiguration(
                    newFbAddress, requestConfiguration);
                auto const ldFeedbackOffset = hwLayer.GetLdFeedbackOffset();
                ldPatches.push_back({ ldFeedbackOffset, feedbackBufferOffset, sizeof(uint32_t) });
            }
            break;
        }
        case ScratchpadOperandIndex:
            ldOffset = hwLayer.GetLdIntermediateOutputOffset();
            break;
        default:
            throw GnaException{ Gna2StatusUnknownError };
        }

        ldPatches.push_back({ ldOffset, bufferOffset, sizeof(uint32_t) });
    }
}

void HardwareRequest::updateGmmModeActiveLists(uint32_t layerIndex, uint32_t layerCount)
{
    auto& layerConfigurations = requestConfiguration.LayerConfigurations;
    auto& model = requestConfiguration.Model;
    if (gmmModeActiveLists.find(layerIndex) == gmmModeActiveLists.end())
    {
        gmmModeActiveLists[layerIndex] = false;
        auto lowerBound = layerConfigurations.lower_bound(layerIndex);
        auto upperBound = layerConfigurations.upper_bound(layerIndex + layerCount);
        for (auto it = lowerBound; it != upperBound; ++it)
        {
            auto const & layer = model.GetLayer(it->first);
            if (it->second->ActList && INTEL_GMM == layer.Operation)
            {
                gmmModeActiveLists[layerIndex] = true;
                break;
            }
        }
    }
}
