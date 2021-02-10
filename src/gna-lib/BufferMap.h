/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Address.h"

#include <map>

namespace GNA
{

using BufferMapBase = std::map<uint32_t, BaseAddress>;

class BufferMap : public BufferMapBase
{
public:
    using map::map;

    BufferMap() = default;
    BufferMap(const BaseAddress& inputBuffer, const BaseAddress& outputBuffer);
};

}

