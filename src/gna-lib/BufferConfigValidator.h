/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Layer.h"

namespace GNA
{

/**
 Stores nullptr buffers location to protect from inferencing and exporting
   models without fully provided data buffers
 */
class BufferConfigValidator
{
public:
    BufferConfigValidator() = default;
    ~BufferConfigValidator() = default;
    BufferConfigValidator(const BufferConfigValidator &) = default;
    BufferConfigValidator(BufferConfigValidator &&) = default;
    BufferConfigValidator& operator=(const BufferConfigValidator&) = default;
    BufferConfigValidator& operator=(BufferConfigValidator&&) = default;

    void populate(Layer const & layer, uint32_t operationIndex);

    void addValidBuffer(uint32_t operationIndex, uint32_t operandIndex);

    void validate() const;

protected:
    std::vector<std::pair<uint32_t /* operationIndex */, uint32_t /* operandIndex */>> missingBuffers;
};

}
