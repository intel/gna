/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Device.h"

#include <cstdint>
#include <memory>

namespace GNA
{

/**
 Used for SW, HW and hybrid inference.
 Acts as virtual device when HW GNA device is NOT present
 and as hardware (hybrid) device when HW GNA device IS present.
 */
class HybridDevice final : public Device
{
public:
    // unique_ptr is guaranteed to be not null
    static std::unique_ptr<HybridDevice> Create(uint32_t index);

    explicit HybridDevice(std::unique_ptr<DriverInterface> && driverInterfaceIn);

    HybridDevice(const HybridDevice&) = delete;
    HybridDevice(HybridDevice&&) = delete;
    HybridDevice& operator=(const HybridDevice&) = delete;
    HybridDevice& operator=(HybridDevice&&) = delete;
    virtual ~HybridDevice() = default;

    void MapMemory(Memory& memoryObject) override;

    void UnMapMemory(Memory & memoryObject) override;

    uint32_t LoadModel(const ApiModel& model) override;

};

}
