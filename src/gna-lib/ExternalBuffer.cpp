/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ExternalBuffer.h"

#include "gna2-model-impl.h"

using namespace GNA;

bool ExternalBuffer::IsSupported(const Gna2Operation&, uint32_t operandIndex)
{
    if (operandIndex == BiasOperandIndex || operandIndex == OutputOperandIndex || operandIndex == InputOperandIndex)
    {
        return true;
    }
    return false;
}
