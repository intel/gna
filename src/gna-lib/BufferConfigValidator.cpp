/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "BufferConfigValidator.h"

using namespace GNA;

void BufferConfigValidator::validate() const
{
    if (!missingBuffers.empty())
    {
        throw GnaException(Gna2StatusNullArgumentNotAllowed);
    }
}

void BufferConfigValidator::populate(Layer const& layer, uint32_t operationIndex)
{
    auto const numberOfOperandsMax = ModelWrapper::GetOperationInfo(layer.OperationNew, NumberOfOperandsMax);
    for (auto i = 0u; i < numberOfOperandsMax; i++)
    {
        auto const operand = layer.TryGetOperand(i);
        if (operand &&
            operand->Mode.Mode != Gna2TensorModeDisabled &&
            operand->Buffer == nullptr)
        {
            missingBuffers.emplace_back(operationIndex, i);
        }
    }
}

void BufferConfigValidator::addValidBuffer(uint32_t operationIndex, uint32_t operandIndex)
{
    auto const found = std::find_if(missingBuffers.begin(), missingBuffers.end(),
        [&operationIndex, &operandIndex](const auto & buffer) { return buffer.first == operationIndex && buffer.second == operandIndex; });
    if (found != missingBuffers.end())
    {
        missingBuffers.erase(found);
    }
}
