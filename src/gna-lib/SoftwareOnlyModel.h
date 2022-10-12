/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "CompiledModel.h"
#include "MemoryContainer.h"
#include "SoftwareModel.h"
#include "Validator.h"

#include "gna2-model-api.h"

#include <cstdint>

namespace GNA
{
class HardwareCapabilities;
class Memory;
class RequestConfiguration;

/**
 Used for Software scoring only in export mode with target/export device
 */
class SoftwareOnlyModel final : public CompiledModel
{
public:
    SoftwareOnlyModel(
        const ApiModel & model,
        const AccelerationDetector& detectorIn,
        const HardwareCapabilities& hwCapabilitiesIn);

    virtual ~SoftwareOnlyModel() = default;

    SoftwareOnlyModel(const SoftwareOnlyModel&) = delete;
    SoftwareOnlyModel(SoftwareOnlyModel&&) = delete;
    SoftwareOnlyModel& operator=(const SoftwareOnlyModel&) = delete;
    SoftwareOnlyModel& operator=(SoftwareOnlyModel&&) = delete;

protected:
    void BuildHardwareModelForExport();

private:
    void score(ScoreContext & context) override;

    void invalidateRequestConfig(uint32_t configId) const override;

    void validateBuffer(MemoryContainer const & requestAllocations, Memory const & memory) const override;

    bool isFullyHardwareCompatible() override;
};

}
