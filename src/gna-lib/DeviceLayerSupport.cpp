/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "DeviceLayerSupport.h"

#include <algorithm>

using namespace GNA;

constexpr Support FROM_GMM = {
    Gna2DeviceGenerationGmm,
    Gna2DeviceGeneration0_9,
    Gna2DeviceGeneration1_0,
    Gna2DeviceGeneration2_0,
    Gna2DeviceGeneration3_0,
    Gna2DeviceGeneration3_1 };

constexpr Support FROM_0_9 = {
    Gna2DeviceGeneration0_9,
    Gna2DeviceGeneration1_0,
    Gna2DeviceGeneration2_0,
    Gna2DeviceGeneration3_0,
    Gna2DeviceGeneration3_1 };

constexpr Support FROM_1_0_TILL_2_0 = {
     Gna2DeviceGeneration1_0,
    Gna2DeviceGeneration2_0 };

constexpr Support FROM_2_0 = {
    Gna2DeviceGeneration2_0,
    Gna2DeviceGeneration3_0,
    Gna2DeviceGeneration3_1 };

constexpr Support FROM_3_0 = {
    Gna2DeviceGeneration3_0,
    Gna2DeviceGeneration3_1 };

constexpr Support FROM_0_9_AUX = FROM_0_9; // Helper for changes of AUX layers


static const std::map<const nn_operation, const Support> FROM_1_0_GMM =
{
    {INTEL_GMM,                 FROM_GMM},
};

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_RNN_CNN =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_RECURRENT,           FROM_3_0},
    {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_0_9_COPY_TRANSPOSE =
{
    {INTEL_COPY,                FROM_0_9_AUX},
    {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
    {INTEL_INTERLEAVE,          FROM_0_9_AUX},
};

static const std::map<const nn_operation, const Support> FROM_3_0_COPY_TRANSPOSE =
{
    {INTEL_COPY,                FROM_3_0},
    {INTEL_DEINTERLEAVE,        FROM_3_0},
    {INTEL_INTERLEAVE,          FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_RNN_CNN_AUX =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_RECURRENT,           FROM_3_0},
    {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
    {INTEL_COPY,                FROM_0_9_AUX},//FROM_3_0
    {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},//FROM_3_0
    {INTEL_INTERLEAVE,          FROM_0_9_AUX},//FROM_3_0
};

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_CNN =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_RNN_CNN_MB_FALSE =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_RECURRENT,           FROM_3_0},
    {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_AFF_CNN_MB_FALSE =
{
    {INTEL_AFFINE,              FROM_3_0},
    {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
    {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_CNN =
{
    {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_3_0_CNN_MB =
{
    {INTEL_AFFINE_MULTIBIAS,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

static const std::map<const nn_operation, const Support> FROM_2_0_MB_3_0_CNN =
{
    {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
    {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
    {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
};

bool DataConfig::IsOperationSupported(nn_operation operation, DataConfig config, Gna2DeviceGeneration generation)
{
    auto const supportMapIterator = Capabilities().find(config);
    if (supportMapIterator == Capabilities().end())
    {
        return false;
    }

    const auto& supportMap = supportMapIterator->second;
    const auto supportIterator = supportMap.find(operation);
    if (supportIterator == supportMap.end())
    {
        return false;
    }

    auto const generationFound = std::find(supportIterator->second.begin(), supportIterator->second.end(), generation);
    return generationFound != supportIterator->second.end();
}

const std::map<const DataConfig, std::map<const nn_operation, const Support>>& DataConfig::Capabilities()
{
    static const std::map<const DataConfig, std::map<const nn_operation, const Support>> caps =
    {
        // input, weight/filter/mean, bias/covariance, output
        {{Gna2DataTypeUint8, Gna2DataTypeUint8, Gna2DataTypeUint32, Gna2DataTypeUint32},
            FROM_1_0_GMM
        },
        {{Gna2DataTypeUint8, Gna2DataTypeUint16, Gna2DataTypeUint32, Gna2DataTypeUint32},
            FROM_1_0_GMM
        },
        {{Gna2DataTypeInt8, DataMode{}, DataMode{}, Gna2DataTypeInt8},
            FROM_3_0_COPY_TRANSPOSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt8},
            FROM_3_0_AFF_RNN_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt16},
            FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt8, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN_MB_FALSE
        },

        // 2B Input
        {{Gna2DataTypeInt16, DataMode{}, DataMode{}, Gna2DataTypeInt16},
            FROM_0_9_COPY_TRANSPOSE
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt16},
           FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt32, true},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt16},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32},
           FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, true},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt8},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt16},
            FROM_2_0_MB_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt32},
            FROM_3_0_CNN_MB
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, Gna2DataTypeInt32, true},
            FROM_2_0_MB_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt8},
            FROM_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt16},
            FROM_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt32},
           FROM_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, DataMode{}, Gna2DataTypeInt32, true},
            FROM_3_0_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias, Gna2DataTypeInt8},
            {
                {INTEL_AFFINE,              FROM_3_0},
                {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias, Gna2DataTypeInt16},
            {
                {INTEL_AFFINE,              FROM_0_9},
                {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
                {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
                {INTEL_RECURRENT,           FROM_0_9},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias, Gna2DataTypeInt32},
            {
                {INTEL_AFFINE,              FROM_3_0},
                {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias, Gna2DataTypeInt32, true},
             {
                {INTEL_AFFINE,              FROM_0_9},
                {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
                {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt16},
           FROM_3_0_AFF_RNN_CNN_AUX
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16},
            FROM_3_0_AFF_RNN_CNN_AUX
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt8},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt16},
            {
                {INTEL_AFFINE,              FROM_0_9},
                {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
                {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
                {INTEL_RECURRENT,           FROM_0_9},
                {INTEL_CONVOLUTIONAL,       FROM_1_0_TILL_2_0},
                {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
                {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
                {INTEL_COPY,                FROM_0_9_AUX},
                {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
                {INTEL_INTERLEAVE,          FROM_0_9_AUX},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt32},
            FROM_3_0_AFF_CNN
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeInt32, true},
            {
                {INTEL_AFFINE,              FROM_0_9},
                {INTEL_AFFINE_DIAGONAL,     FROM_0_9},
                {INTEL_AFFINE_MULTIBIAS,    FROM_2_0},
                {INTEL_CONVOLUTIONAL,       FROM_1_0_TILL_2_0},
                {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
                {INTEL_CONVOLUTIONAL_2D,    FROM_3_0}
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt8},
             FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt16},
            {
                {INTEL_AFFINE,              FROM_3_0},
                {INTEL_AFFINE_DIAGONAL,     FROM_3_0},
                {INTEL_RECURRENT,           FROM_3_0},
                {INTEL_CONVOLUTIONAL_1D,    FROM_3_0},
                {INTEL_CONVOLUTIONAL_2D,    FROM_3_0},
                {INTEL_COPY,                FROM_0_9_AUX},
                {INTEL_DEINTERLEAVE,        FROM_0_9_AUX},
                {INTEL_INTERLEAVE,          FROM_0_9_AUX},
            }
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt32},
           FROM_3_0_AFF_CNN_MB_FALSE
        },
        {{Gna2DataTypeInt16, Gna2DataTypeInt16, DataMode{}, Gna2DataTypeInt32, true},
            FROM_3_0_AFF_CNN_MB_FALSE
        },
    };
    return caps;
}
