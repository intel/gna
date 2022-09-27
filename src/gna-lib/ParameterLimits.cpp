/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ParameterLimits.h"

#include "Expect.h"
#include "ModelError.h"
#include "Shape.h"

using namespace GNA;

MultiplierLimits::MultiplierLimits(const MultiplierLimits& multipliers, ModelErrorSource error) :
    MultiplierMap{ multipliers },
    Error{ error }
{
}

MultiplierLimits::MultiplierLimits(uint32_t multiplier, ModelErrorSource error) :
    MultiplierMap{ { multiplier } },
    Error( error )
{}

uint32_t& MultiplierLimits::at(Gna2DataType type)
{
    Expect::InRange<Gna2DataType>(type, Gna2DataTypeWeightScaleFactor, Gna2StatusNullArgumentNotAllowed);
    return MultiplierMap::at(static_cast<size_t>(type));
}

uint32_t MultiplierLimits::at(Gna2DataType type) const
{
    Expect::InRange(type, Gna2DataTypeWeightScaleFactor, Gna2StatusNullArgumentNotAllowed);
    return MultiplierMap::at(static_cast<size_t>(type));
}

void MultiplierLimits::SetEffective(DataType type)
{
    if (Gna2DataTypeNone != type &&
        (*this)[type] != 0)
    {
        (*this)[Gna2DataTypeNone] = at(type);
    }
}

uint32_t MultiplierLimits::GetEffective() const
{
    return at(Gna2DataTypeNone);
}
