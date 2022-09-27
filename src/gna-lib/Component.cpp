/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Component.h"

#include "Capabilities.h"
#include "Expect.h"
#include "Layout.h"
#include "ParameterLimits.h"

using namespace GNA;

Component::Component(const Shape & dimensions, uint32_t componentIndex, bool isParameter) :
    Dimensions{ dimensions },
    Count{ Dimensions.GetNumberOfElements() },
    ComponentIndex{ componentIndex },
    IsParameter{ isParameter },
    validator{ nullptr }
{
}

Component::Component(const Component & component, const Validator & validatorIn, bool validateDimensions,
    uint32_t componentIndex, bool isParameter) :
    Dimensions{ component.Dimensions },
    Count{ component.Count },
    ComponentIndex{ componentIndex },
    IsParameter{ isParameter }
{
    validator = std::make_unique<const Validator>(validatorIn);
    Expect::NotNull(validator);
    Validate(*validator->Capabilities, validateDimensions);
}

Component::Component(const Shape& dimensions, const Validator& validatorIn, bool validateDimensions,
    uint32_t componentIndex, bool isParameter) :
    Component{ Component{ dimensions.Reshape(validatorIn.Order), componentIndex, isParameter }, validatorIn, validateDimensions, componentIndex, isParameter }
{}

ModelValue Component::at(char dimension) const
{
    if (!IsParameter)
    {
        return Dimensions.AsModelValue(dimension, ComponentIndex);
    }
    auto dimensionValue = Dimensions.AsModelValue(dimension, Gna2DisabledU32);
    dimensionValue.SetParameter(ComponentIndex);
    dimensionValue.Source.Type = Gna2ItemTypeShapeDimensions;
    return dimensionValue;
}

void Component::Validate(const FullCapabilitiesMap& caps, nn_operation operation) const
{
    auto const validatorOpt = LayerValidator{ *validator, operation };
    auto const & limits = caps.GetLatestCaps(validatorOpt);
    Validate(*limits, true);
}

void Component::ValidateDimensions(DataType type) const
{
    // update Multiplier when varies for data modes
    auto caps = *validator->Capabilities;
    for (auto & dim : caps.Dimensions)
    {
        dim.second.Multipliers.SetEffective(type);
    }
    Validate(caps, true);
}

void Component::Validate(const ComponentLimits& limits, bool validateDimensions) const
{
    if (validateDimensions)
    {
        Expect::Equal(Dimensions.LayoutOrder.operator _tensor_order(),
            limits.Order.Value, Gna2StatusXnnErrorLyrInvalidTensorOrder);
        ExpectShapeIsValid(limits.Dimensions);
    }
}

// If any dimension in map is invalid prints error status code and throws exception.
void Component::DimensionIsValid(const Shape::value_type& dimension, const RangeLimits<uint32_t>& limits) const
{
    auto const dimIndex = static_cast<uint32_t>(Dimensions.LayoutOrder.GetApiIndex(dimension.first));
    auto const shapeDim = Dimensions.LayoutOrder.at(dimIndex);
    auto const dim = at(shapeDim);
    ModelErrorHelper::ExpectBelowEq(dim, limits.Max.Value);
    ModelErrorHelper::ExpectAboveEq(dim, limits.Min.Value);

    ModelErrorHelper::ExpectMultiplicityOf(dim, limits.Multipliers.GetEffective());
}

void Component::ExpectShapeIsValid(const ShapeLimits& limits) const
{
    for (const auto& dim : Dimensions)
    {
        auto const limit = limits.at(dim.first);
        DimensionIsValid(dim, limit);
    }
}