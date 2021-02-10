/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "GnaException.h"
#include "ParameterLimits.h"

#include "gna-api-types-xnn.h"

#include <gna2-common-api.h>
#include <gna2-model-impl.h>

#include <cstdint>
#include <map>

namespace GNA
{

struct DataMode
{
    template<typename T, typename DT>
    static T ToSize(const DT mode)
    {
        try
        {
            return static_cast<T>(GetSizes<DT>().at(mode));
        }
        catch (const std::exception&)
        {
            throw GnaException(Gna2StatusDataModeInvalid);
        }
    }

    DataMode() = delete;
    DataMode(const DataMode&) = default;
    DataMode(const gna_data_mode dataMode);
    DataMode(const uint32_t dataMode);
    DataMode(const DataType dataType, const TensorMode tensorMode = Gna2TensorModeDefault);
    ~DataMode() = default;

    operator gna_data_mode() const
    {
        return Value;
    }

    const gna_data_mode Value;

    const DataType Type;

    const TensorMode Mode;

    // Size on data element in bytes
    const uint32_t Size;

protected:
    template<typename T>
    static const std::map<const T, const uint32_t>& GetSizes();

    static DataType TypeFromDataMode(const gna_data_mode dataMode);
    static TensorMode ModeFromDataMode(const gna_data_mode dataMode);
    static gna_data_mode ModeFromDataMode(const DataType dataType);
};

bool operator ==(const gna_data_mode &left, const DataMode &right);

bool operator !=(const gna_data_mode &left, const DataMode &right);

bool operator ==(Gna2DataType left, const DataMode &right);

bool operator !=(Gna2DataType left, const DataMode &right);

using DataModeLimits = SetLimits<DataMode>;

}
