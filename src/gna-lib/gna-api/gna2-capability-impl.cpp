/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "gna2-capability-impl.h"

#include "ApiWrapper.h"
#include "Expect.h"

using namespace GNA;

GNA2_API enum Gna2Status Gna2GetLibraryVersion(char * versionBuffer, uint32_t versionBufferSize)
{
    static const char versionString[] = GNA_LIBRARY_VERSION_STRING;

    const std::function<Gna2Status()> command = [&]()
    {
        GNA::Expect::NotNull(versionBuffer);
        GNA::Expect::True(static_cast<uint64_t>(32) <= versionBufferSize, Gna2StatusMemorySizeInvalid);
        GNA::Expect::True(sizeof(versionString) <= static_cast<uint64_t>(versionBufferSize), Gna2StatusMemorySizeInvalid);
        const auto reqSize = snprintf(versionBuffer, versionBufferSize, "%s", versionString);
        GNA::Expect::True(reqSize >= 0 && static_cast<unsigned>(reqSize) + 1 <= versionBufferSize,
            Gna2StatusMemorySizeInvalid);
        return Gna2StatusSuccess;
    };
    return GNA::ApiWrapper::ExecuteSafely(command);
}
