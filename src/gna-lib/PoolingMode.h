/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "PoolingKernelArguments.h"

#include "GnaException.h"

#include "gna2-model-api.h"

#include <stdexcept>

namespace GNA
{

inline KernelPoolingMode ToKernelPoolingMode(Gna2PoolingMode const apiMode)
{
    static const std::map<Gna2PoolingMode, KernelPoolingMode> poolingMap{
       { Gna2PoolingModeMax, KernelPoolingModeMax },
       { Gna2PoolingModeSum, KernelPoolingModeSum },
       { Gna2PoolingModeDisabled, KernelPoolingModeNone }
    };
    try
    {
        return poolingMap.at(apiMode);
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusCnnErrorPoolType);
    }
}

}
