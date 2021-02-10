/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "HardwareCapabilities.h"
#include "ParameterLimits.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cstddef>
#include <functional>

namespace GNA
{

class FullCapabilitiesMap;

// Functor for validating if buffer is within memory boundaries
using ValidBoundariesFunctor = std::function<void(const void *, size_t)>;

class BaseValidator
{
public:
    BaseValidator(
        const HardwareCapabilities hwCapabilities,
        const ValidBoundariesFunctor validBoundariesIn);
    virtual ~BaseValidator() = default;

    void ValidateBuffer(const void* const buffer, size_t size, const uint32_t alignment) const;

    inline void ValidateBufferIfSet(const void* const buffer, size_t size,
        const AlignLimits& alignLimits = {GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid}) const
    {
        if (buffer)
        {
            ValidateBuffer(buffer, size, alignLimits.Value);
        }
    }

    const HardwareCapabilities HwCapabilities;

protected:
    const ValidBoundariesFunctor bufferValidator;
};

class LayerValidator : public BaseValidator
{
public:
    LayerValidator(const BaseValidator& validator, nn_operation operation);
    virtual ~LayerValidator() = default;

    const nn_operation Operation;
};

class Validator : public LayerValidator
{
public:
    Validator(const LayerValidator& validator, const FullCapabilitiesMap& capabilities);
    virtual ~Validator() = default;

    const ComponentLimits * const Capabilities;
    const gna_tensor_order Order;
};

}

