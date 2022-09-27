/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Memory.h"

#include "DeviceManager.h"
#include "DriverInterface.h"
#include "Expect.h"
#include "GnaException.h"
#include "gna2-memory-impl.h"
#include "KernelArguments.h"

using namespace GNA;

// just makes object from arguments
Memory::Memory(void* bufferIn, uint32_t userSize, uint32_t alignment) :
    Address{ bufferIn },
    size{ RoundUp(userSize, alignment) },
    allocationOwner{ false }
{
}

// allocates and zeros memory
Memory::Memory(const uint32_t userSize, uint32_t alignment) :
    size{ RoundUp(userSize, alignment) }
{
    Expect::InRange(size, 1u, GNA_MAX_MEMORY_FOR_SINGLE_ALLOC, Gna2StatusMemorySizeInvalid);
    buffer = _gna_malloc(size);
    Expect::ValidBuffer(buffer);
    memset(buffer, 0, size); // this is costly and probably not needed
}

Memory::~Memory()
{
    if (buffer != nullptr && allocationOwner)
    {
        if (mapped)
        {
            try
            {
                DeviceManager::Get().UnmapMemoryFromAllDevices(*this);
            }
            catch (...)
            {
                Log->Error("UnmapMemoryFromAllDevices failed.\n");
            }
            mapped = false;
            id = 0;
        }

        _gna_free(buffer);
        buffer = nullptr;
        size = 0;
    }
}

void Memory::Map(DriverInterface& ddi)
{
    if (mapped || !allocationOwner)
    {
        throw GnaException(Gna2StatusUnknownError);
    }

    id = ddi.MemoryMap(buffer, size);

    mapped = true;
}
void Memory::Unmap(DriverInterface& ddi)
{
    if (mapped)
    {
        ddi.MemoryUnmap(id);
        mapped = false;
    }
}

uint64_t Memory::GetId() const
{
    if (!mapped)
    {
        throw GnaException(Gna2StatusUnknownError);
    }

    return id;
}

void Memory::SetTag(uint32_t newTag)
{
    tag = newTag;
}

Gna2MemoryTag Memory::GetMemoryTag() const
{
    return static_cast<Gna2MemoryTag>(tag);
}
