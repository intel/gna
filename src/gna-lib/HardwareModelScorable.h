/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Address.h"
#include "HardwareModel.h"
#include "HardwareRequest.h"
#include "IScorable.h"
#include "MemoryContainer.h"

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
        const HardwareCapabilities& hwCapsIn, const std::vector<std::unique_ptr<SubModel>>& subModelsIn);
    virtual ~HardwareModelScorable() = default;

    HardwareModelScorable(const HardwareModelScorable&) = delete;
    HardwareModelScorable(HardwareModelScorable&&) = delete;
    HardwareModelScorable& operator=(const HardwareModelScorable&) = delete;
    HardwareModelScorable& operator=(HardwareModelScorable&&) = delete;

    void InvalidateConfig(uint32_t configId);

    void Score(ScoreContext & context) override;

    uint32_t GetBufferOffsetForConfiguration(
        const BaseAddress& address,
        const RequestConfiguration& requestConfiguration) const;

    void ValidateConfigBuffer(MemoryContainer const & requestAllocations,
        Memory const & bufferMemory) const;

protected:
    DriverInterface &driverInterface;

    std::map<uint32_t, std::unique_ptr<HardwareRequest>> hardwareRequests;
    std::mutex hardwareRequestsLock;

    const std::vector<std::unique_ptr<SubModel>>& subModels;

    void prepareAllocationsAndModel() override;

    bool IsSoftwareLayer(uint32_t layerIndex) const override;

    std::unique_ptr<Memory> allocLD(uint32_t ldMemorySize, uint32_t ldSize = Memory::GNA_BUFFER_ALIGNMENT) override;
};

}
