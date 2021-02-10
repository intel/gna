/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "BufferMap.h"
#include "ModelWrapper.h"

using namespace GNA;

BufferMap::BufferMap(const BaseAddress& inputBuffer, const BaseAddress& outputBuffer) :
    map()
{
    if (inputBuffer)
        emplace(InputOperandIndex, inputBuffer);
    if (outputBuffer)
        emplace(OutputOperandIndex, outputBuffer);
}
