/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "igemv8.h"
#include "KernelArguments.h"

#include <cstdint>

void TransposeKernelImpl1B(TransposeConfig const * const transposeConfig)
{
    uint32_t i, j;
    for (i = 0; i < transposeConfig->rowCount; i++)
    {
        for (j = 0; j < transposeConfig->columnCount; j++)
        {
            ((int8_t*)transposeConfig->output)[j * transposeConfig->rowCount + i] = ((int8_t*)transposeConfig->input)[i * transposeConfig->columnCount + j];
        }
    }
}
