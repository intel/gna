/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "PoolingKernelArguments.h"

PoolingConfig::PoolingConfig(PoolingConfig const * const source, int64_t * const bufferIn) :
    PoolingConfig{ *source }
{
    Buffer = bufferIn;
}

PoolingConfig::PoolingConfig(KernelPoolingMode const mode, uint32_t const sizeIn, uint32_t const stepIn) :
    Mode{ mode },
    Size{ sizeIn },
    Step{ stepIn },
    Buffer{ nullptr }
{
}
