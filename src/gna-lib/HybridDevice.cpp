/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "HybridDevice.h"

#include "DriverInterface.h"
#include "HybridModel.h"

#include <cstdint>
#include <memory>

using namespace GNA;

std::unique_ptr<HybridDevice> HybridDevice::Create(uint32_t index)
{
    auto device = std::make_unique<HybridDevice>(DriverInterface::Create(index));
    Expect::NotNull(device);
    return device;
}

HybridDevice::HybridDevice(std::unique_ptr<DriverInterface> && driverInterfaceIn) :
    Device(std::make_unique<HardwareCapabilitiesDevice>(driverInterfaceIn->GetCapabilities()))
{
    driverInterface = std::move(driverInterfaceIn);
}

void HybridDevice::MapMemory(Memory & memoryObject)
{
    if (hardwareCapabilities->IsHardwareSupported())
    {
        memoryObject.Map(*driverInterface);
    }
}

void HybridDevice::UnMapMemory(Memory & memoryObject)
{
    if (hardwareCapabilities->IsHardwareSupported())
    {
        memoryObject.Unmap(*driverInterface);
    }
}

uint32_t HybridDevice::LoadModel(const ApiModel& model)
{
    auto compiledModel = std::make_unique<HybridModel>(model, accelerationDetector, *hardwareCapabilities, *driverInterface);

    return StoreModel(std::move(compiledModel));
}
