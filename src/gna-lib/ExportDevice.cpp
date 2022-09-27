/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ExportDevice.h"
#include "HardwareModelNoMMU.h"
#include "HardwareModelSue1.h"
#include "SoftwareOnlyModel.h"

#include <cstdint>
#include <memory>

using namespace GNA;

ExportDevice::ExportDevice(Gna2DeviceVersion targetDeviceVersion) :
    Device(std::make_unique<HardwareCapabilitiesExport>(targetDeviceVersion))
{}

uint32_t ExportDevice::LoadModel(const ApiModel& model)
{
    auto compiledModel = std::make_unique<SoftwareOnlyModel>(model, accelerationDetector, *hardwareCapabilities);

    return StoreModel(std::move(compiledModel));
}

void* GNA::Dump(CompiledModel const & model, Gna2ModelSueCreekHeader* modelHeader, Gna2Status* status, Gna2UserAllocator customAlloc)
{
    // Validate parameters
    Expect::NotNull(status);
    Expect::NotNull(modelHeader);
    Expect::NotNull(customAlloc);

    // creating HW layer descriptors directly into dump memory
    auto hwModel = std::make_unique<HardwareModelSue1>(model, customAlloc);
    if (!hwModel)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }
    auto const address = hwModel->Export();
    if (!address)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }

    hwModel->PopulateHeader(*modelHeader);

    *status = Gna2StatusSuccess;
    return address;
}

void GNA::DumpComponentNoMMu(CompiledModel const & model, Gna2UserAllocator customAlloc,
    void *& exportData, uint32_t & exportDataSize, const Gna2ModelExportComponent component,
    const Gna2DeviceVersion targetDevice)
{
    Expect::NotNull(customAlloc);

    // creating HW layer descriptors directly into dump memory
    auto hwModel = std::make_unique<HardwareModelNoMMU>(model, customAlloc, targetDevice);
    if (!hwModel)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }

    hwModel->ExportComponent(exportData, exportDataSize, component);
}
