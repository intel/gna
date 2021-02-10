/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "MemoryContainer.h"

#include "Expect.h"
#include "GnaException.h"
#include "Memory.h"

#include <algorithm>
#include <string.h>

using namespace GNA;

MemoryContainerElement::MemoryContainerElement(Memory const& memoryIn, uint32_t notAlignedIn, uint32_t pageAlignedIn) :
    memory{ memoryIn },
    notAligned{ notAlignedIn },
    pageAligned{ pageAlignedIn }
{
}

MemoryContainerElement::operator Memory const&() const
{
    return memory.get();
}

void * MemoryContainerElement::GetBuffer() const
{
    return memory.get().GetBuffer();
}

uint32_t MemoryContainerElement::GetSize() const
{
    return memory.get().GetSize();
}

inline uint32_t MemoryContainerElement::GetNotAligned() const
{
    return notAligned;
}

inline uint32_t MemoryContainerElement::GetPageAligned() const
{
    return pageAligned;
}

inline void MemoryContainerElement::ResetOffsets(uint32_t notAlignedIn, uint32_t pageAlignedIn)
{
    notAligned = notAlignedIn;
    pageAligned = pageAlignedIn;
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
        totalMemorySizeAlignedToPage += RoundUp(value.GetSize(), PAGE_SIZE);
        totalMemorySize += value.GetSize();
    }
}

MemoryContainer::const_iterator MemoryContainer::FindByAddress(BaseAddress const& address) const
{
    auto const foundIt = std::find_if(cbegin(), cend(),
        [&address](auto const & memory)
    {
        return address.InRange(memory.GetBuffer(), memory.GetSize());
    });

    return foundIt;
}

bool MemoryContainer::Contains(const void* buffer, const size_t bufferSize) const
{
    auto const & memory = FindByAddress(buffer);
    if (cend() != memory &&
        Expect::InMemoryRange(buffer, bufferSize, memory->GetBuffer(), memory->GetSize()))
    {
        return true;
    }
    return false;
}

uint32_t MemoryContainer::GetBufferOffset(const BaseAddress& address, uint32_t alignment, uint32_t initialOffset) const
{
    auto const foundIt = FindByAddress(address);
    if (cend() != foundIt)
    {
        auto const internalOffset = address.GetOffset(BaseAddress{ foundIt->GetBuffer() });
        if (1 == alignment)
        {
            return initialOffset + foundIt->GetNotAligned() + internalOffset;
        }
        if (PAGE_SIZE == alignment)
        {
            return initialOffset + foundIt->GetPageAligned() + internalOffset;
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
        auto const memorySize = memory.GetSize();
        auto const memoryBuffer = memory.GetBuffer();
        memcpy_s(address, destinationSize, memoryBuffer, memorySize);
        destinationSize -= memorySize;
        address += memorySize;
    }
}

void MemoryContainer::WriteData(FILE* file) const
{
    Expect::NotNull(file);
    for (const auto & memory : *this)
    {
        auto const memorySize = memory.GetSize();
        auto const memoryBuffer = memory.GetBuffer();
        fwrite(memoryBuffer, memorySize, sizeof(uint8_t), file);
    }
}

void MemoryContainer::invalidateOffsets()
{
    uint32_t offset = 0;
    uint32_t offsetPageAligned = 0;
    //for (auto && memoryIter = cbegin(); memoryIter != cend(); ++memoryIter)
    for (auto & memoryIter : *this)
    {
        memoryIter.ResetOffsets(offset, offsetPageAligned);

        auto const memorySize = memoryIter.GetSize();
        offset += memorySize;
        offsetPageAligned += RoundUp(memorySize, PAGE_SIZE);
    }
}
