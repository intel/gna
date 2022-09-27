/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#ifndef __GNA2_COMMON_IMPL_H
#define __GNA2_COMMON_IMPL_H

#define NOMINMAX 1

#include "gna2-common-api.h"

#include <map>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace GNA
{

typedef enum Gna2DeviceVersion DeviceVersion;

/**
 Version of device that is used by default by GNA Library in software mode,
 when no hardware device is available.

 @note
 Usually it will be the latest existing GNA device (excluding embedded)
 on the time of publishing the library, value may change with new release.
 */
#define GNA2_DEFAULT_DEVICE_VERSION Gna2DeviceVersion3_0

constexpr auto DefaultDeviceVersion = DeviceVersion{ GNA2_DEFAULT_DEVICE_VERSION };

typedef enum Gna2Status ApiStatus;

constexpr auto Gna2DisabledU32 = uint32_t(GNA2_DISABLED);

/** Size of memory alignment for data tensors */
constexpr auto GNA_MEM_ALIGN = uint32_t{ 64 };

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

    static std::string ToString(Gna2Status statusIn);
};

extern const std::map<Gna2Status, std::string>& staticDestructionProtectionHelper;

template<typename Key, typename Value>
static Value GetMappedOrDefault(Key key, Value defaultValue, const std::map<Key, Value>& map)
{
    const auto found = map.find(key);
    if (found != map.end())
    {
        return found->second;
    }
    return defaultValue;
}

template<class Container, typename Key>
bool contains(const Container & container, const Key & key)
{
    return container.end() != container.find(key);
}

}
#endif //ifndef __GNA2_COMMON_IMPL_H
