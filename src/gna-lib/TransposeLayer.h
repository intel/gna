/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Layer.h"

#include "KernelArguments.h"
#include "XnnKernel.h"

#include "common.h"

#include <cstdint>
#include <map>

namespace GNA
{
class BaseValidator;
struct LayerConfiguration;

// Transpose Layer descriptor converter
class TransposeLayer : public Layer
{
public:
    TransposeLayer(const nn_layer& layer, const BaseValidator& validatorIn);
    TransposeLayer(const Gna2Operation& apiOperation, const BaseValidator& validatorIn);
    virtual ~TransposeLayer() = default;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

protected:
    virtual DataConfig GetDataMode() const override;

private:
    void computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const;
    void compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const;

    const KernelMap<TransposeKernel>& transposeKernels;
    std::unique_ptr<TransposeConfig> transposeHiddenConfig;
};

}

