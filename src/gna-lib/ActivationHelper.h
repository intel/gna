/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "gna2-model-api.h"

namespace GNA
{

struct ActivationHelper
{
    static bool IsEnabled(const Gna2Operation& apiOperation);
    static void ExpectProper(const Gna2Tensor& activation);
};

}
