/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "DriverInterface.h"
#include "SubModel.h"

namespace GNA
{

class SoftwareModel;
class Memory;
class AccelerationDetector;
class Layer;
struct LayerConfiguration;
class RequestConfiguration;
class RequestProfiler;

struct ScoreContext
{
    ScoreContext(uint32_t layerIndexIn, uint32_t layerCountIn,
        RequestConfiguration& requestConfigurationIn, RequestProfiler &profilerIn, KernelBuffers *buffersIn) :
        subModelType{ Software },
        layerIndex{ layerIndexIn },
        layerCount{ layerCountIn },
        requestConfiguration{ requestConfigurationIn },
        profiler{ profilerIn },
        buffers{ buffersIn },
        saturationCount{ 0 }
    {}

    SubModelType subModelType;
    uint32_t layerIndex;
    uint32_t layerCount;
    RequestConfiguration& requestConfiguration;
    RequestProfiler &profiler;
    KernelBuffers *buffers;
    uint32_t saturationCount;

    void Update(SubModel const * const subModel)
    {
        subModelType = subModel->Type;
        layerIndex = subModel->LayerIndex;
        layerCount = subModel->GetLayerCount();
    }
};

class IScorable
{
public:
    virtual ~IScorable() = default;
    virtual void Score(ScoreContext & context) = 0;
};

}
