/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "HardwareCapabilities.h"
#include "ParameterLimits.h"


#include <cstddef>
#include <functional>

namespace GNA
{

class FullCapabilitiesMap;

// Functor for validating if buffer is within memory boundaries
using ValidBoundariesFunctor = std::function<void(const void *, size_t, uint32_t alignment)>;

class BaseValidator
{
public:
    BaseValidator(
        Gna2DeviceGeneration generation,
        const ValidBoundariesFunctor validBoundariesIn);
    virtual ~BaseValidator() = default;


    inline void ValidateBufferIfSet(const void* const buffer, size_t size,
        const AlignLimits& alignLimits = { GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid }) const
    {
        if (buffer)
        {
            validateBuffer(buffer, size, alignLimits.Value);
        }
    }

    const Gna2DeviceGeneration Generation;

protected:
    void validateBuffer(const void* const buffer, size_t size, const uint32_t alignment) const;

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
    Validator(const LayerValidator& validator, const FullCapabilitiesMap& capabilities,
        bool isBufferOptional = false);
    virtual ~Validator() = default;

    void ValidateBuffer(const void* const buffer, size_t size, const uint32_t alignment) const;

    const ComponentLimits * const Capabilities;
    const gna_tensor_order Order;
    const bool IsBufferOptional;
};

}

