/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "CompiledModel.h"
#include "HardwareModelScorable.h"
#include "MemoryContainer.h"
#include "SoftwareModel.h"
#include "SubModel.h"
#include "Validator.h"

#include "gna2-common-impl.h"
#include "gna2-model-api.h"

#include <cstdint>
#include <memory>
#include <vector>

struct KernelBuffers;

namespace GNA
{
class DriverInterface;
class HardwareCapabilities;
class Layer;
class Memory;
class RequestConfiguration;
struct LayerConfiguration;

/**
 Hybrid = Software (relaxed limits) + Hardware (present device limits) + Software (present device limits)
 */
class HybridModel final : public CompiledModel
{
public:
    HybridModel(
        const ApiModel & model,
        const AccelerationDetector& detectorIn,
        const HardwareCapabilities& hwCapabilitiesIn,
        DriverInterface &ddi);

    virtual ~HybridModel() = default;

    HybridModel(const HybridModel&) = delete;
    HybridModel(HybridModel&&) = delete;
    HybridModel& operator=(const HybridModel&) = delete;
    HybridModel& operator=(HybridModel&&) = delete;

private:
    SoftwareModel & GetSoftwareModel() override
    {
        if (softwareModelForPresentDevice)
        {
            return *softwareModelForPresentDevice;
        }
        return softwareModel;
    }

    SoftwareModel const & GetSoftwareModel() const override
    {
        if (softwareModelForPresentDevice)
        {
            return *softwareModelForPresentDevice;
        }
        return softwareModel;
    }

    // software model for present hardware device to maintain consistency
    // used only for building hardware model
    std::unique_ptr<SoftwareModel> softwareModelForPresentDevice = {};

    std::unique_ptr<HardwareModelScorable> hardwareModel;

    bool fullyHardwareCompatible = false;

    bool verifyFullyHardwareCompatible();

    void BuildHardwareModel(DriverInterface &ddi);

    const std::vector<std::unique_ptr<SubModel>>& getSubModels();

    void createSubModels(const HardwareCapabilities& hwCaps);

    static bool shouldSplit(SubModel & currentSubModel, SubModelType nextSubModelType, const HardwareCapabilities& hwCaps);

    SubModelType getSubModelType(
        const HardwareCapabilities &hwCaps, uint32_t layerIndex) const;

    void score(ScoreContext & context) override;

    void ScoreSubModel(ScoreContext & context);

    void ScoreHwSubModel(ScoreContext & context);

    std::map<DeviceVersion,
        std::vector<std::unique_ptr<SubModel>>> subModels = {};

    void invalidateRequestConfig(uint32_t configId) const override;

    void validateBuffer(MemoryContainer const & requestAllocations, Memory const & memory) const override;

    bool isFullyHardwareCompatible() override;

    bool shouldUseSoftwareMode(RequestConfiguration& config) const;

    DeviceVersion getSoftwareConsistencyDeviceVersion() const;
};

}
