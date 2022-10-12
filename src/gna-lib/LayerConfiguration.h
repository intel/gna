/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "ActiveList.h"
#include "BufferMap.h"
#include "KernelArguments.h"

#include <memory>

namespace GNA
{

struct KernelConfigs
{
    std::unique_ptr<const AffineConfig> Affine;
    std::unique_ptr<RecurrentConfig> Recurrent;
    std::unique_ptr<const ConvolutionConfig> Convolution;
    std::unique_ptr<TransposeConfig> Transpose;
    std::unique_ptr<CopyConfig> Copy;
    std::unique_ptr<GmmConfig> Gmm;
};

struct LayerConfiguration
{
    std::unique_ptr<ActiveList> ActList;
    BufferMap Buffers;
    KernelConfigs Configs;
    std::unique_ptr<BaseConfig> ConfigList[TransformOperationCount];

    void EmplaceBuffer(uint32_t operandIndex, void * address);
    void RemoveBuffer(uint32_t operandIndex);
};
}
