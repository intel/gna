/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "ParameterLimits.h"

#include <gna2-model-impl.h>

#include <cstdint>

namespace GNA
{

struct DataMode
{
    constexpr DataMode() :
        Type{ Gna2DataTypeNone },
        Mode{ Gna2TensorModeDisabled },
        Size{ 0 }
    {}
    constexpr DataMode(const DataMode &) = default;
    constexpr DataMode(DataMode &&) = default;
    DataMode(DataType type);
    DataMode(DataType type, TensorMode tensorMode);
    ~DataMode() = default;

    constexpr bool operator<(const DataMode & mode) const
    {
        if (Type != mode.Type)
        {
            return Type < mode.Type;
        }
        return Mode <= mode.Mode;
    }

    constexpr bool operator!=(const DataMode & mode) const
    {
        return Type != mode.Type || Mode != mode.Mode;
    }

    constexpr bool operator==(const DataMode & mode) const
    {
        return !(operator!=(mode));
    }

    DataMode &operator =(const DataMode & mode) = default;

    DataType Type;
    TensorMode Mode;
    // Size on data element in bytes
    uint32_t Size;

protected:
    static uint32_t GetSize(DataType type);

    static TensorMode ModeFromType(DataType type);
    static DataType TypeFromMode(DataType type, TensorMode mode);
};

using DataModeLimits = SetLimits<DataMode>;

}
