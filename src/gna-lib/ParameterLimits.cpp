/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "ParameterLimits.h"

#include "ModelError.h"
#include "Shape.h"

using namespace GNA;

// If any dimension in map is invalid prints error status code and throws exception.
template<typename T>
inline static void DimensionIsValid(const T& dimension, const RangeLimits<T>& limits)
{
    ModelErrorHelper::ExpectBelowEq(dimension, limits.Max.Value, Gna2ItemTypeShapeDimensions);
    ModelErrorHelper::ExpectAboveEq(dimension, limits.Min.Value, Gna2ItemTypeShapeDimensions);

    ModelErrorHelper::ExpectMultiplicityOf(dimension, limits.Multipliers.GetEffective(), Gna2ItemTypeShapeDimensions);
}

void GNA::ExpectShapeIsValid(const Shape& dimensions, const ShapeLimits& limits)
{
    for (const auto& dim : dimensions)
    {
        try
        {
            auto limit = limits.at(dim.first);
            DimensionIsValid(dim.second, limit);
        }
        catch (GnaModelErrorException& e)
        {
            const auto index = dimensions.LayoutOrder.GetApiIndex(dim.first);
            e.SetDimensionIndex(index);
            throw;
        }
    }
}
