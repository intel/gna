/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Capabilities.h"
#include "Component.h"
#include "OperationConfig.h"
#include "Transform.h"
#include "XnnKernel.h"

#include <memory>

namespace GNA
{
class FullCapabilitiesMap;
template<typename T> struct SetLimits;

class PoolingFunction2D : public Transform<PoolingConfig2D, PoolingKernel2D>
{
public:
    static std::unique_ptr<PoolingFunction2D> Create(
        const TransformFactoryConfig& config,
        const OperationConfig& operation);

    PoolingFunction2D(const BaseTransformConfig<PoolingKernel2D>& config,
        KernelPoolingMode mode, std::unique_ptr<const Component> window,
        std::unique_ptr<const Component> stride);

    ~PoolingFunction2D() = default;

    const KernelPoolingMode Mode;

    std::unique_ptr<const Component> Window;

    std::unique_ptr<const Component> Stride;

protected:
    static std::unique_ptr<PoolingFunction2D> create(
        const TransformFactoryConfig& config,
        const OperationConfig& operation);

    virtual void updateExecutionKernelConfig(ExecutionKernelConfig<PoolingConfig2D> & config)
        const override
    {
        setSoftwareScratchPad(config);
    }
};

}
