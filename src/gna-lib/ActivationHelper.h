/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "gna2-model-api.h"
#include "gna-api-types-xnn.h"

namespace GNA
{

class ActivationHelper
{
    ActivationHelper() = delete;
public:
    static bool IsEnabled(const Gna2Operation& apiOperation);
    static void ExpectProper(const Gna2Tensor& activation);

    static bool IsEnabled(const intel_convolutional_layer_t& cnnDetails);
    static bool IsEnabled(const intel_pwl_func_t& pwl);

    static intel_pwl_func_t const& GetPwl(void const *layerDetails, gna_layer_operation operation);
};

}
