/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Shape.h"
#include "Validator.h"


#include <memory>

namespace GNA
{
struct ComponentLimits;

struct Component
{
    Component(const Shape& dimensions, uint32_t componentIndex, bool isParameter);

    Component(const Component& component, const Validator& validator, bool validateDimensions,
        uint32_t componentIndex, bool isParameter = true);

    Component(const Shape& dimensions, const Validator& validator, bool validateDimensions,
        uint32_t componentIndex, bool isParameter = true);

    virtual ~Component() = default;

    /**
     * Gets Dimensions value at dimension key
     */
    inline uint32_t at(const gna_tensor_dim dimension) const
    {
        return Dimensions.at(dimension);
    }

    ModelValue at(char dimension) const;

    Shape Dimensions;

    // Total number of elements
    uint32_t Count;

    const uint32_t ComponentIndex;

    const bool IsParameter;

    void Validate(const FullCapabilitiesMap & caps, nn_operation operation) const;

    void ValidateDimensions(DataType type) const;

    void ExpectShapeIsValid(const ShapeLimits& limits) const;

protected:
    void Validate(const ComponentLimits& limits, bool validateDimensions = true) const;
    void DimensionIsValid(const Shape::value_type& dimension, const RangeLimits<uint32_t>& limits) const;

    std::unique_ptr<const Validator> validator;
};

}
