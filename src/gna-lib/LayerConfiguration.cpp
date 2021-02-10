/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "LayerConfiguration.h"

using namespace GNA;

void LayerConfiguration::EmplaceBuffer(uint32_t operandIndex, void *address)
{
    Buffers.emplace(operandIndex, address);
}

void LayerConfiguration::RemoveBuffer(uint32_t operandIndex)
{
    Buffers.erase(operandIndex);
}
