/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "PoolingKernelArguments.h"

#include "GnaException.h"

#include "gna-api-types-xnn.h"
#include "gna2-model-api.h"

#include <stdexcept>

namespace GNA
{

class PoolingMode
{
public:
    template<class T>
    PoolingMode(const T type) try:
        mode{toPoolingMode(type)}
    {
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusCnnErrorPoolType);
    }
    operator KernelPoolingMode() const
    {
        return mode;
    }

private:
    static KernelPoolingMode toPoolingMode(intel_pool_type_t const legacyType);
    static KernelPoolingMode toPoolingMode(Gna2PoolingMode const apiMode);
    const KernelPoolingMode mode;
};

}
