/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Shape.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"

#include <memory>

namespace GNA
{
struct ComponentLimits;

struct Component
{
    Component(const Shape& dimensions);

    Component(const Component& component, const Validator& validator, bool validateDimensions = true);

    Component(const Shape& dimensions, const Validator& validator, bool validateDimensions = true);

    virtual ~Component() = default;

    nn_operation GetEffectiveOperationType() const;

    /**
     * Gets Dimensions value at dimension key
     */
    inline uint32_t at(const gna_tensor_dim dimension) const
    {
        return Dimensions.at(dimension);
    }

    virtual ModelValue AsModelValue(char dimension) const;

    Shape Dimensions;

    // Total number of elements
    uint32_t Count;

    void Validate(const FullCapabilitiesMap & caps, nn_operation operation) const;

protected:
    void Validate(const ComponentLimits& limits, bool validateDimensions = true) const;

    std::unique_ptr<const Validator> validator;
};

}
