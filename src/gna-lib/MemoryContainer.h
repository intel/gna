/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Address.h"

#include <cstdint>
#include <map>
#include <vector>

namespace GNA
{

class Memory;

class MemoryContainerElement : protected std::reference_wrapper<Memory const>
{
public:
    MemoryContainerElement(Memory const& memoryIn, uint32_t notAlignedIn, uint32_t pageAlignedIn);

    using std::reference_wrapper<Memory const>::operator const type&;
    Memory const& get() const;

    const uint32_t NotAligned;
    const uint32_t PageAligned;
};

using MemoryContainerType = std::vector< MemoryContainerElement >;

class MemoryContainer : public MemoryContainerType
{
public:
    void Append(MemoryContainer const & source);

    void Emplace(Memory const & value);

    const_iterator FindByAddress(BaseAddress const & address) const;

    bool Contains(const void *buffer, const size_t bufferSize = 1) const;

    uint32_t GetMemorySize() const
    {
        return static_cast<uint32_t>(totalMemorySize);
    }

    uint32_t GetMemorySizeAlignedToPage() const
    {
        return static_cast<uint32_t>(totalMemorySizeAlignedToPage);
    }

    uint32_t GetBufferOffset(const BaseAddress& address, uint32_t alignment = 1, uint32_t initialOffset = 0) const;

    template<typename T>
    void CopyEntriesTo(std::vector<T> & destination) const;

    void CopyData(void * destination, size_t destinationSize) const;

protected:
    uint32_t totalMemorySizeAlignedToPage = 0;

    uint32_t totalMemorySize = 0;
};

template <typename T>
void MemoryContainer::CopyEntriesTo(std::vector<T> & destination) const
{
    for (auto const & memory : *this)
    {
        destination.emplace_back(memory);
    }
}

}
