/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "gna2-model-api.h"

#include <cstdint>

namespace GNA
{
class ExternalBuffer
{
public:
    static bool IsSupported(const Gna2Operation& operation, uint32_t operandIndex);
};

}
