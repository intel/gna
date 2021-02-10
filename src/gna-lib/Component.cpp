/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Component.h"

#include "Capabilities.h"
#include "Expect.h"
#include "Layout.h"
#include "ParameterLimits.h"

using namespace GNA;

Component::Component(const Shape & dimensions) :
    Dimensions{ dimensions },
    Count{ Dimensions.GetNumberOfElements() },
    validator{ nullptr }
{
}

Component::Component(const Component & component, const Validator & validatorIn, bool validateDimensions) :
    Dimensions{ component.Dimensions },
    Count{ component.Count }
{
    validator = std::make_unique<const Validator>(validatorIn);
    Expect::NotNull(validator.get());
    Validate(*validator->Capabilities, validateDimensions);
}

Component::Component(const Shape& dimensions, const Validator& validatorIn, bool validateDimensions) :
    Component{ Component{ dimensions.Reshape(validatorIn.Order) }, validatorIn, validateDimensions }
{}

nn_operation Component::GetEffectiveOperationType() const
{
    if (validator)
    {
        return validator->Operation;
    }
    return LAYER_OPERATION_TYPE_COUT;
}

void Component::Validate(const FullCapabilitiesMap& caps, nn_operation operation) const
{
    auto const validatorOpt = LayerValidator{ *validator, operation };
    auto const & limits = caps.GetLatestCaps(validatorOpt);
    Validate(*limits, true);
}

void Component::Validate(const ComponentLimits& limits, bool validateDimensions) const
{
    if (validateDimensions)
    {
        Expect::Equal(Dimensions.LayoutOrder.operator _tensor_order(),
            limits.Order.Value, Gna2StatusXnnErrorLyrInvalidTensorOrder);
        GNA::ExpectShapeIsValid(Dimensions, limits.Dimensions);
    }
}

ModelValue Component::AsModelValue(char dimension) const
{
    return Dimensions.AsModelValue(dimension);
}
