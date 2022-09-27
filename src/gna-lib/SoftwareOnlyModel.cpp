/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#define NOMINMAX 1

#include "SoftwareOnlyModel.h"

using namespace GNA;

SoftwareOnlyModel::SoftwareOnlyModel(const ApiModel& model, const AccelerationDetector& detectorIn,
    const HardwareCapabilities& hwCapabilitiesIn) :
    CompiledModel{ model, detectorIn, hwCapabilitiesIn, hwCapabilitiesIn.GetDeviceVersion() }
{
    BuildHardwareModelForExport();
}

void SoftwareOnlyModel::BuildHardwareModelForExport()
{
    auto const verifyModel = std::make_unique<HardwareModelTarget>(*this, hwCapabilities);
    if (!verifyModel)
    {
        throw GnaException(Gna2StatusResourceAllocationError);
    }
    // discard HardwareModelTarget as only needed for build verification
}

void SoftwareOnlyModel::score(ScoreContext& context)
{
    softwareModel.Score(context);
}

void SoftwareOnlyModel::invalidateRequestConfig(uint32_t configId) const
{
    UNREFERENCED_PARAMETER(configId);
}

void SoftwareOnlyModel::validateBuffer(MemoryContainer const & requestAllocations, Memory const & memory) const
{
    UNREFERENCED_PARAMETER(requestAllocations);
    UNREFERENCED_PARAMETER(memory);
}

bool SoftwareOnlyModel::isFullyHardwareCompatible()
{
    return false;
}
