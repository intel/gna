/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Layer.h"
#include "XnnKernel.h"

namespace GNA
{

class CopyLayer : public Layer
{
public:
    CopyLayer(const nn_layer& layer, const BaseValidator& validatorIn);

    CopyLayer(const Gna2Operation& operation, const BaseValidator& validatorIn);

    virtual ~CopyLayer() = default;
    void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

    const uint32_t ColumnCount;
    const uint32_t RowCount;

protected:
    virtual DataConfig GetDataMode() const override;

    static Shape GetCopyShape(const Gna2Operation& operation);

private:
    void computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const;
    void compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const;

    const KernelMap<CopyKernel>& copyKernels;
    CopyConfig copyHiddenConfig;
    static const FullCapabilitiesMap limits;
};

}
