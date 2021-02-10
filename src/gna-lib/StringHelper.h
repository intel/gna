/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Expect.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>

namespace GNA
{
struct StringHelper
{
    static void Copy(char& dstBegin, const uint32_t dstSize, const std::string& source);

    template <class T>
    static uint32_t GetMaxLength(const std::map<T, std::string>& container)
    {
        uint32_t maxLen = 0;
        for (const auto & s : container)
        {
            maxLen = (std::max)(maxLen, static_cast<uint32_t>(s.second.size()));
        }
        return maxLen + 1;
    }

    template <class T>
    static const std::string& GetFromMap(const std::map<T, std::string>& container, T containerKey)
    {
        const auto found = container.find(containerKey);
        GNA::Expect::True(found != container.end(), Gna2StatusIdentifierInvalid);
        return found->second;
    }
};

}
