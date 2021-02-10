/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "PoolingMode.h"

#include "gna-api-types-xnn.h"

#include <map>

using namespace GNA;

KernelPoolingMode PoolingMode::toPoolingMode(intel_pool_type_t const legacyType)
{
    static const std::map<intel_pool_type_t, KernelPoolingMode> poolingModeMap{
        { INTEL_NO_POOLING, KernelPoolingModeNone },
        { INTEL_SUM_POOLING, KernelPoolingModeSum },
        { INTEL_MAX_POOLING, KernelPoolingModeMax },
    };
    return poolingModeMap.at(legacyType);
}

KernelPoolingMode PoolingMode::toPoolingMode(Gna2PoolingMode const apiMode)
{
    static const std::map<Gna2PoolingMode, KernelPoolingMode> poolingMap{
        { Gna2PoolingModeMax, KernelPoolingModeMax },
        { Gna2PoolingModeSum, KernelPoolingModeSum },
        { Gna2PoolingModeDisabled, KernelPoolingModeNone }
    };
    return poolingMap.at(apiMode);
}
