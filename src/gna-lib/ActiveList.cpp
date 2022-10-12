/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ActiveList.h"
#include "Expect.h"

using namespace GNA;

std::unique_ptr<ActiveList> ActiveList::Create(const ActiveList& activeList)
{
    Expect::ValidBuffer(activeList.Indices);
    return std::make_unique<ActiveList>(activeList);
}

ActiveList::ActiveList(const uint32_t indicesCountIn, const uint32_t* indicesIn) :
    IndicesCount{indicesCountIn},
    Indices{indicesIn} { }

