/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Capabilities.h"
#include "Expect.h"
#include "ParameterLimits.h"
#include "Validator.h"

#include <cstddef>

using namespace GNA;

BaseValidator::BaseValidator(
    Gna2DeviceGeneration generation,
    const ValidBoundariesFunctor validBoundariesIn) :
    Generation{ generation },
    bufferValidator{ validBoundariesIn }
{
}

void BaseValidator::validateBuffer(const void * const buffer, size_t size, const uint32_t alignment) const
{
    bufferValidator(buffer, size, alignment);
}

LayerValidator::LayerValidator(const BaseValidator& validator, nn_operation operation) :
    BaseValidator{ validator },
    Operation{ operation }
{
}

Validator::Validator(const LayerValidator & validator, const FullCapabilitiesMap & capabilities,
    bool isBufferOptional) :
    LayerValidator{ validator },
    Capabilities{ capabilities.GetLatestCaps(validator) },
    Order{ Capabilities->Order.Value },
    IsBufferOptional{ isBufferOptional }
{
}

void Validator::ValidateBuffer(const void* const buffer, size_t size, const uint32_t alignment) const
{
    if (IsBufferOptional)
    {
        ValidateBufferIfSet(buffer, size, { alignment, Gna2StatusMemoryAlignmentInvalid });
    }
    else
    {
        validateBuffer(buffer, size, alignment);
    }
}
