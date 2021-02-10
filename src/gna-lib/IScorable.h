/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "DriverInterface.h"

namespace GNA
{

class SoftwareModel;
class Memory;
class AccelerationDetector;
class Layer;
struct LayerConfiguration;
class RequestConfiguration;
class RequestProfiler;

class IScorable
{
public:
    virtual ~IScorable() = default;
    virtual uint32_t Score(
        uint32_t layerIndex,
        uint32_t layerCount,
        const RequestConfiguration& requestConfiguration,
        RequestProfiler *profiler,
        KernelBuffers *buffers) = 0;
};

}
