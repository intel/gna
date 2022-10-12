/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include <cstdint>
#include <memory>

namespace GNA
{

// Request's active list configuration and data
struct ActiveList
{
    static std::unique_ptr<ActiveList> Create(const ActiveList& activeList);

    ActiveList(const uint32_t indicesCountIn, const uint32_t* indicesIn);
    ActiveList(const ActiveList& activeList) = default;
    ActiveList(ActiveList &&) = default;
    ActiveList& operator=(const ActiveList&) = delete;

    const uint32_t IndicesCount;
    const uint32_t* Indices;
};

}
