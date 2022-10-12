/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Device.h"

namespace GNA
{

class ExportDevice final : public Device
{
public:
    explicit ExportDevice(Gna2DeviceVersion targetDeviceVersion);

    ExportDevice(const ExportDevice&) = delete;
    ExportDevice(ExportDevice&&) = delete;
    ExportDevice& operator=(const ExportDevice&) = delete;
    ExportDevice& operator=(ExportDevice&&) = delete;
    virtual ~ExportDevice() = default;

    uint32_t LoadModel(const ApiModel& model) override;
};

void* Dump(CompiledModel const & model, Gna2ModelSueCreekHeader* modelHeader, Gna2Status* status,
    Gna2UserAllocator customAlloc);

void DumpComponentNoMMu(CompiledModel const & model, Gna2UserAllocator customAlloc,
    void *& exportData, uint32_t & exportDataSize, Gna2ModelExportComponent component,
    Gna2DeviceVersion targetDevice);

}
