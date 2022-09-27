/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "StringHelper.h"

#include "Expect.h"

#include <cstdint>

using namespace GNA;

void GNA::StringHelper::Copy(char& dstBegin, const uint32_t dstSize, const std::string& source)
{
    GNA::Expect::True(source.size() + 1 <= dstSize, Gna2StatusMemorySizeInvalid);
    const auto reqSize = snprintf(&dstBegin, dstSize, "%s", source.c_str());
    GNA::Expect::True(reqSize >= 0 && static_cast<unsigned>(reqSize) + 1 <= dstSize,
        Gna2StatusMemorySizeInvalid);
}
