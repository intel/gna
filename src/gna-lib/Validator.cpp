/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Capabilities.h"
#include "Expect.h"
#include "ModelError.h"
#include "ParameterLimits.h"
#include "Validator.h"

#include "common.h"
#include "gna-api.h"

#include <cstddef>

using namespace GNA;

BaseValidator::BaseValidator(
    const HardwareCapabilities hwCapabilities,
    const ValidBoundariesFunctor validBoundariesIn) :
    HwCapabilities{ hwCapabilities },
    bufferValidator{ validBoundariesIn }
{
}

void BaseValidator::ValidateBuffer(const void * const buffer, size_t size, const uint32_t alignment) const
{
    ModelErrorHelper::ExpectNotNull(buffer);
    ModelErrorHelper::ExpectBufferAligned(buffer, alignment);
    bufferValidator(buffer, size);
}

LayerValidator::LayerValidator(const BaseValidator& validator, nn_operation operation) :
    BaseValidator{ validator },
    Operation{ operation }
{
}

Validator::Validator(const LayerValidator & validator, const FullCapabilitiesMap & capabilities) :
    LayerValidator{ validator },
    Capabilities{ capabilities.GetLatestCaps(validator) },
    Order{ Capabilities->Order.Value }
{
}
