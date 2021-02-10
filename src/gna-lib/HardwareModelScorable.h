/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Address.h"
#include "HardwareModel.h"
#include "HardwareRequest.h"
#include "IScorable.h"
#include "MemoryContainer.h"

#include "KernelArguments.h"

#include "gna-api.h"

#include <cstdint>
#include <memory>
#include <map>
#include <vector>

namespace GNA
{

class DriverInterface;
class HardwareCapabilities;
class Layer;
class RequestConfiguration;
class RequestProfiler;

class HardwareModelScorable : public HardwareModel, public IScorable
{
public:

    HardwareModelScorable(CompiledModel const & softwareModel, DriverInterface &ddi,
        const HardwareCapabilities& hwCapsIn);
    virtual ~HardwareModelScorable() = default;

    void InvalidateConfig(uint32_t configId);

    virtual uint32_t Score(
        uint32_t layerIndex,
        uint32_t layerCount,
        const RequestConfiguration& requestConfiguration,
        RequestProfiler *profiler,
        KernelBuffers *buffers) override;

    uint32_t GetBufferOffsetForConfiguration(
        const BaseAddress& address,
        const RequestConfiguration& requestConfiguration) const;

    void ValidateConfigBuffer(MemoryContainer const & requestAllocations,
        Memory const & bufferMemory) const;

protected:
    DriverInterface &driverInterface;

    std::map<uint32_t, std::unique_ptr<HardwareRequest>> hardwareRequests;
    std::mutex hardwareRequestsLock;

    virtual void prepareAllocationsAndModel() override;
};

}
