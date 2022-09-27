/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Transform.h"

#include "ActivationHelper.h"
#include "OperationConfig.h"

#include <set>

using namespace GNA;

bool TransformFactoryConfig::HasMandatoryActivation() const
{
    return mandatoryActivation;
}

bool TransformFactoryConfig::IsActivationNotSupported() const
{
    static const std::set<nn_operation> forbiddenActivation{
        GNA_LAYER_CNN_2D_POOLING,
        INTEL_INTERLEAVE,
        INTEL_COPY,
        INTEL_DEINTERLEAVE };
    return 0 < forbiddenActivation.count(validator.Operation);
}

Gna2Tensor TransformFactoryConfig::GetActivation() const
{
    return activation;
}

void TransformFactoryConfig::InitActivation(const Gna2Operation & operation)
{
    mandatoryActivation = HasMandatoryActivation(operation);
    activation = GetActivation(operation);
}

inline bool TransformFactoryConfig::HasMandatoryActivation(const Gna2Operation & operation)
{
    if(OperationConfig::IsCNN1D(operation))
    {
        return Gna2PoolingModeDisabled != OperationConfig::GetPoolingMode(operation);
    }
    return operation.Type == Gna2OperationTypeRecurrent;
}

inline Gna2Tensor TransformFactoryConfig::GetActivation(const Gna2Operation & operation)
{
    if (operation.NumberOfOperands > PwlOperandIndex && operation.Type != Gna2OperationTypeGmm &&
        operation.Operands != nullptr && operation.Operands[PwlOperandIndex] != nullptr)
    {
        return *operation.Operands[PwlOperandIndex];
    }
    Gna2Tensor disabled{};
    disabled.Mode = Gna2TensorModeDisabled;
    return disabled;
}

Tensor const & BaseTransform::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case InputOperandIndex:
        if (Input)
        {
            return *Input;
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case OutputOperandIndex:
        return GetOperandIfExistOrThrow(Output);
    default:
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
}
