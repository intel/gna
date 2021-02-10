/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GNA2_INFERENCE_IMPL_H
#define __GNA2_INFERENCE_IMPL_H

#include "gna2-common-impl.h"
#include "gna-api.h"
#include "../gna-api/gna2-inference-api.h"

#include <string>
#include <vector>
#include <cstdint>


namespace GNA
{
    /**
 * List of all supported acceleration modes
 */
class AccelerationMode
{
public:
    AccelerationMode(Gna2AccelerationMode basicMode, bool hardwareConsistencyEnabled = false);

    bool IsHardwareEnforced() const;

    bool IsSoftwareEnforced() const;

    // operator needed by std::map
    bool operator<(const AccelerationMode& right) const;

    AccelerationMode GetEffectiveSoftwareAccelerationMode(const std::vector<Gna2AccelerationMode>& supportedCpuAccelerations) const;

    static void ExpectValid(Gna2AccelerationMode mode);

    void SetMode(Gna2AccelerationMode newMode);

    Gna2AccelerationMode GetMode() const;

    void SetHwConsistency(bool consistencyEnabled);
    bool GetHwConsistency() const;

    const char* GetName() const;

private:
    Gna2AccelerationMode mode;

    static const char* UNKNOWN_ACCELERATION_MODE_NAME;

    bool hardwareConsistency = false;

    void enforceValidity();
};

}

#endif // __GNA2_INFERENCE_IMPL_H
