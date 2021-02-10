/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Memory.h"

#include "common.h"
#include "DeviceManager.h"
#include "DriverInterface.h"
#include "Expect.h"
#include "GnaException.h"
#include "KernelArguments.h"

using namespace GNA;

// just makes object from arguments
Memory::Memory(void* bufferIn, uint32_t userSize, uint32_t alignment) :
    Address{ bufferIn },
    size{ RoundUp(userSize, alignment) }
{
    deallocate = false;
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
    if (buffer != nullptr && deallocate)
    {
        if (mapped)
        {
            DeviceManager::Get().UnMapMemoryFromAll(*this);
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
    if (mapped)
    {
        throw GnaException(Gna2StatusUnknownError);
    }

    id = ddi.MemoryMap(buffer, size);

    mapped = true;
}
void Memory::Unmap(DriverInterface& ddi)
{
    if (!mapped)
    {
        throw GnaException(Gna2StatusUnknownError);
    }
    ddi.MemoryUnmap(id);
    mapped = false;
}

uint64_t Memory::GetId() const
{
    if (!mapped)
    {
        throw GnaException(Gna2StatusUnknownError);
    }

    return id;
}
