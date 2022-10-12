/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "MemoryContainer.h"

#include "Expect.h"
#include "GnaException.h"
#include "gna2-memory-impl.h"
#include "Macros.h"
#include "Memory.h"

#include <algorithm>

using namespace GNA;

MemoryContainerElement::MemoryContainerElement(Memory const& memoryIn, uint32_t notAlignedIn, uint32_t pageAlignedIn) :
    std::reference_wrapper<Memory const>{memoryIn},
    NotAligned{ notAlignedIn },
    PageAligned{ pageAlignedIn }
{
}

Memory const& MemoryContainerElement::get() const
{
    return std::reference_wrapper<Memory const>::get();
}

void MemoryContainer::Append(MemoryContainer const & source)
{
    for (auto const & value : source)
    {
        Emplace(value);
    }
}

void MemoryContainer::Emplace(Memory const & value)
{
    if (!Contains(value, value.GetSize()))
    {
        emplace_back(value, totalMemorySize, totalMemorySizeAlignedToPage);
        totalMemorySizeAlignedToPage += RoundUp(value.GetSize(), MemoryBufferAlignment);
        totalMemorySize += value.GetSize();
    }
}

MemoryContainer::const_iterator MemoryContainer::FindByAddress(BaseAddress const& address) const
{
    auto const foundIt = std::find_if(cbegin(), cend(),
        [&address](auto const & memory)
    {
        return address.InRange(memory.get().GetBuffer(), memory.get().GetSize());
    });

    return foundIt;
}

bool MemoryContainer::Contains(const void* buffer, const size_t bufferSize) const
{
    auto const & memory = FindByAddress(buffer);
    return cend() != memory &&
        Expect::InMemoryRange(buffer, bufferSize,
        (*memory).get().GetBuffer(), (*memory).get().GetSize());
}

uint32_t MemoryContainer::GetBufferOffset(const BaseAddress& address, uint32_t alignment, uint32_t initialOffset) const
{
    auto const foundIt = FindByAddress(address);
    if (cend() != foundIt)
    {
        auto const internalOffset = address.GetOffset(BaseAddress{ (*foundIt).get().GetBuffer() });
        if (1 == alignment)
        {
            return initialOffset + foundIt->NotAligned + internalOffset;
        }
        if (MemoryBufferAlignment == alignment)
        {
            return initialOffset + foundIt->PageAligned + internalOffset;
        }
    }
    return 0;
}

void MemoryContainer::CopyData(void* destination, size_t destinationSize) const
{
    if (destinationSize < totalMemorySize)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }

    auto address = static_cast<uint8_t *>(destination);
    for (const auto & memory : *this)
    {
        auto const memorySize = memory.get().GetSize();
        auto const memoryBuffer = memory.get().GetBuffer();
        memcpy_s(address, destinationSize, memoryBuffer, memorySize);
        destinationSize -= memorySize;
        address += memorySize;
    }
}
