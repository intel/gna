/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "HardwareLayer.h"

#include "DriverInterface.h"
#include "MemoryContainer.h"
#include "RequestConfiguration.h"

#include "gna2-instrumentation-api.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include "HardwareModel.h"

namespace GNA
{

class HardwareModelScorable;
class Layer;
class RequestConfiguration;
struct LayerConfiguration;

enum GnaOperationMode : uint8_t
{
    GMM = 0,
    xNN = 1
};

class HardwareRequest
{
public:
    HardwareRequest(const HardwareModelScorable& hwModelIn,
        const RequestConfiguration& requestConfigurationIn,
        MemoryContainer const & modelAllocations);

    void Invalidate();
    void Update(uint32_t layerIndex, uint32_t layerCount, GnaOperationMode mode);

    bool IsSwFallbackEnabled() const
    {
        return requestConfiguration.Acceleration.IsSoftwareFallbackEnabled();
    }

    /* these fields will not change between request executions */
    const uint8_t HwPerfEncoding;
    const uint32_t RequestConfigId;

    /* these fields can change on each request execution */
    GnaOperationMode Mode;

    /* xNN fields */
    uint32_t LayerBase;
    uint32_t LayerCount;

    /* GMM fields */
    uint32_t GmmOffset;
    bool GmmModeActiveListOn;

    std::vector<DriverBuffer> DriverMemoryObjects;

    /* Driver specific request data*/
    std::unique_ptr<uint8_t[]> CalculationData;
    size_t CalculationSize;

    /* Hardware request ready for driver submition indicator */
    bool SubmitReady = false;

    ProfilerConfiguration* GetProfilerConfiguration() const
    {
        return requestConfiguration.GetProfilerConfiguration();
    }

private:

    const RequestConfiguration& requestConfiguration;
    const HardwareModelScorable& hwModel;

    std::map<uint32_t, bool> gmmModeActiveLists;

    void updateGmmModeActiveLists(uint32_t layerIndex, uint32_t layerCount);

    void generateBufferPatches(const LayerConfiguration& layerConfiguration,
        const Layer &layer, const HardwareLayer &hwLayer);
};

}
