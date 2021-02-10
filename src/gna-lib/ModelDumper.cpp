/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Device.h"

#include "CompiledModel.h"
#include "DataMode.h"
#include "Expect.h"
#include "HardwareModelNoMMU.h"
#include "HardwareModelSue1.h"
#include "Layer.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Memory.h"

#include "gna-api-status.h"
#include "gna-api.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>

using namespace GNA;

void* Device::Dump(uint32_t modelId, Gna2ModelSueCreekHeader* modelHeader, Gna2Status* status, Gna2UserAllocator customAlloc)
{
    // Validate parameters
    Expect::NotNull(status);
    Expect::NotNull(modelHeader);
    Expect::NotNull(reinterpret_cast<void *>(customAlloc));

    auto const & model = *models.at(modelId);

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

void Device::DumpLdNoMMu(uint32_t modelId, Gna2UserAllocator customAlloc,
    void *& exportData, uint32_t & exportDataSize)
{
    Expect::NotNull(reinterpret_cast<void *>(customAlloc));
    auto const & model = *models.at(modelId);

    // creating HW layer descriptors directly into dump memory
    auto hwModel = std::make_unique<HardwareModelNoMMU>(model, customAlloc);
    if (!hwModel)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }
    const auto& allocations = model.GetAllocations();
    hwModel->ROBeginAddress = allocations.begin()->GetBuffer();


    hwModel->ExportLd(exportData, exportDataSize);
    Expect::NotNull(exportData, Gna2StatusResourceAllocationError);
}