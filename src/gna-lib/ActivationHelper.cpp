/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ActivationHelper.h"

#include "ModelWrapper.h"

using namespace GNA;

bool ActivationHelper::IsEnabled(const Gna2Operation & apiOperation)
{
    return ModelWrapper::HasEnabledOperand(apiOperation, PwlOperandIndex);
}

void ActivationHelper::ExpectProper(const Gna2Tensor & activation)
{
    ModelErrorHelper::ExpectInSet(activation.Type, { Gna2DataTypePwlSegment });
    ModelErrorHelper::ExpectInSet(activation.Mode, { Gna2TensorModeDefault });
    ModelErrorHelper::ExpectEqual(activation.Shape.NumberOfDimensions, 1, Gna2ItemTypeShapeNumberOfDimensions);
}
