/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#ifndef __GNA2_MEMORY_IMPL_H
#define __GNA2_MEMORY_IMPL_H

#include <cstdint>

template<typename T>
constexpr T GnaCeilDiv(T number, T divider)
{
    return (number + divider - 1) / divider;
}

template<typename T>
constexpr T RoundUp(T number, T significance)
{
    if (0 == significance)
    {
        return number;
    }
    else
    {
        return GnaCeilDiv(number, significance) * significance;
    }
}

#endif // __GNA2_MEMORY_IMPL_H
