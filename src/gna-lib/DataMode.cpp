/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "DataMode.h"

#include "GnaException.h"
#include "ModelError.h"

#include "gna2-model-api.h"
#include "gna2-model-impl.h"

#include <cstdint>

using namespace GNA;

uint32_t DataMode::GetSize(DataType type)
{
    try
    {
        static const std::map<const DataType, const uint32_t> sizes =
        {
            {Gna2DataTypeNone, 0},
            {Gna2DataTypeBoolean, 1},
            {Gna2DataTypeInt4, 1},
            {Gna2DataTypeInt8, 1},
            {Gna2DataTypeInt16, 2},
            {Gna2DataTypeInt32, 4},
            {Gna2DataTypeInt64, 8},
            {Gna2DataTypeUint4, 1},
            {Gna2DataTypeUint8, 1},
            {Gna2DataTypeUint16, 2},
            {Gna2DataTypeUint32, 4},
            {Gna2DataTypeUint64, 8},
            {Gna2DataTypeCompoundBias, 8},
            {Gna2DataTypePwlSegment, 8},
            {Gna2DataTypeWeightScaleFactor, 8},
        };

        return sizes.at(type);
    }
    catch (const std::exception&)
    {
        throw GnaModelErrorException(Gna2ItemTypeOperandType, Gna2ErrorTypeNotInSet, type);
    }
}

TensorMode DataMode::ModeFromType(DataType type)
{
    try
    {
        static const std::map<const DataType, const TensorMode> types =
        {
            {Gna2DataTypeNone, Gna2TensorModeDisabled},
            {Gna2DataTypeBoolean, Gna2TensorModeDefault},
            {Gna2DataTypeInt4,  Gna2TensorModeDefault},
            {Gna2DataTypeInt8,  Gna2TensorModeDefault},
            {Gna2DataTypeInt16, Gna2TensorModeDefault},
            {Gna2DataTypeInt32, Gna2TensorModeDefault},
            {Gna2DataTypeInt64, Gna2TensorModeDefault},
            {Gna2DataTypeUint4, Gna2TensorModeDefault},
            {Gna2DataTypeUint8, Gna2TensorModeDefault},
            {Gna2DataTypeUint16, Gna2TensorModeDefault},
            {Gna2DataTypeUint32, Gna2TensorModeDefault},
            {Gna2DataTypeUint64, Gna2TensorModeDefault},
            {Gna2DataTypeCompoundBias, Gna2TensorModeDefault},
            {Gna2DataTypePwlSegment, Gna2TensorModeDefault},
            {Gna2DataTypeWeightScaleFactor, Gna2TensorModeDefault},
        };
        return types.at(type);
    }
    catch (const std::exception&)
    {
        throw GnaModelErrorException(Gna2ItemTypeOperandType, Gna2ErrorTypeNotInSet, type);
    }
}

DataType DataMode::TypeFromMode(DataType type, TensorMode mode)
{
    ModelErrorHelper::ExpectInSet(mode,
        { Gna2TensorModeDefault, Gna2TensorModeExternalBuffer, Gna2TensorModeDisabled },
        Gna2ItemTypeOperandMode);
    switch (mode)
    {
    case Gna2TensorModeDisabled:
        return Gna2DataTypeNone;
    case Gna2TensorModeConstantScalar:
        return Gna2DataTypeInt4;
    default:
        return type;
    }
}

DataMode::DataMode(DataType type) :
    Type{ type },
    Mode{ ModeFromType(type) },
    Size{ GetSize(Type) }
{
}

DataMode::DataMode(DataType type, TensorMode tensorMode) :
    Type{ TypeFromMode(type, tensorMode) },
    Mode{ tensorMode },
    Size{ GetSize(Type) }
{
}
