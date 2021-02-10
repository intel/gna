/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GNA2_COMMON_IMPL_H
#define __GNA2_COMMON_IMPL_H

#include "gna-api.h"
#include "gna2-common-api.h"

#include <map>
#include <stdint.h>
#include <string>
#include <unordered_map>

namespace GNA
{

typedef enum Gna2DeviceVersion DeviceVersion;

/**
 Version of device that is used by default by GNA Library in software mode,
 when no hardware device is available.

 @see
 Gna2RequestConfigEnableHardwareConsistency() to change hardware device
 version in software mode.

 @note
 Usually it will be the latest existing GNA device (excluding embedded)
 on the time of publishing the library, value may change with new release.
 */
#define GNA2_DEFAULT_DEVICE_VERSION Gna2DeviceVersion2_0

DeviceVersion const DefaultDeviceVersion = GNA2_DEFAULT_DEVICE_VERSION;

typedef enum Gna2Status ApiStatus;

constexpr uint32_t const Gna2DisabledU32 = (uint32_t)GNA2_DISABLED;

constexpr int32_t const Gna2DisabledI32 = (int32_t)GNA2_DISABLED;

constexpr uint32_t const Gna2DefaultU32 = (uint32_t)GNA2_DEFAULT;

constexpr int32_t const Gna2DefaultI32 = (int32_t)GNA2_DEFAULT;

constexpr uint32_t const Gna2NotSupportedU32 = (uint32_t)GNA2_NOT_SUPPORTED;

constexpr int32_t const Gna2NotSupportedI32 = (int32_t)GNA2_NOT_SUPPORTED;

/* Workaround for old compilers that do not handle enums as map keys */
struct EnumHash
{
    template<typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

Gna2DeviceVersion Gna2GetVersionForLegacy(gna_device_version legacyVersion);

template<typename T>
inline Gna2DeviceVersion Gna2DeviceVersionFromInt(T value)
{
    static_assert(sizeof(Gna2DeviceVersion) <= sizeof(T), "");
    union DeviceVersionOrInt
    {
        T number;
        Gna2DeviceVersion version;
    } out;
    out.number = value;
    return out.version;
}

class StatusHelper
{
public:

    static const std::map<Gna2Status, std::string>& GetStringMap();
    static const std::map<Gna2Status, std::string>& GetDescriptionMap();

    static std::string ToString(Gna2Status statusIn);
};

}
#endif //ifndef __GNA2_COMMON_IMPL_H
