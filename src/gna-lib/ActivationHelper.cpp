/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "ActivationHelper.h"

#include "Expect.h"
#include "ModelWrapper.h"
#include "Transform.h"

#include "common.h"

using namespace GNA;

bool ActivationHelper::IsEnabled(const Gna2Operation & apiOperation)
{
    return ModelWrapper::HasEnabledOperand(apiOperation, PwlOperandIndex);
}

void ActivationHelper::ExpectProper(const Gna2Tensor & activation)
{
    ModelErrorHelper::ExpectInSet(activation.Type, { Gna2DataTypePwlSegment });
    ModelErrorHelper::ExpectInSet(activation.Mode, { Gna2TensorModeDefault });
    ModelErrorHelper::ExpectNotNull(activation.Data, Gna2ItemTypeOperandData, PwlOperandIndex);
    ModelErrorHelper::ExpectEqual(activation.Shape.NumberOfDimensions, 1, Gna2ItemTypeShapeNumberOfDimensions);
}

bool ActivationHelper::IsEnabled(const nn_layer_conv & cnnDetails)
{
    return IsEnabled(cnnDetails.pwl);
}

bool ActivationHelper::IsEnabled(const intel_pwl_func_t & pwl)
{
    return nullptr != pwl.pSegments && pwl.nSegments > 0;
}

const nn_func_pwl& ActivationHelper::GetPwl(void const *layerDetails, nn_operation operation)
{
    Expect::NotNull(layerDetails, Gna2StatusXnnErrorLyrOperation);
    switch (operation)
    {
    case INTEL_AFFINE: /* FALLTHRU */
    case INTEL_AFFINE_DIAGONAL:
        return static_cast<nn_layer_affine const*>(layerDetails)->pwl;
    case INTEL_AFFINE_MULTIBIAS:
        return static_cast<nn_layer_affine_multi const*>(layerDetails)->pwl;
    case INTEL_CONVOLUTIONAL:
        return static_cast<nn_layer_conv const*>(layerDetails)->pwl;
    case INTEL_CONVOLUTIONAL_2D:
        return static_cast<nn_layer_cnn2d const*>(layerDetails)->activation;
    case INTEL_RECURRENT:
        return static_cast<nn_layer_recurrent const*>(layerDetails)->pwl;
    default:
        throw GnaException{ Gna2StatusXnnErrorLyrOperation };
    }
}
