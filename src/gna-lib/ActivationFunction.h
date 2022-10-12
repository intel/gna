/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "DataMode.h"
#include "KernelArguments.h"
#include "Tensor.h"
#include "Transform.h"
#include "XnnKernel.h"
#include "pwl.h"

#include <cstdint>
#include <memory>

namespace GNA
{

class FullCapabilitiesMap;

class ActivationFunction : public Transform<ActivationConfig, ActivationKernel>
{
public:
    static std::unique_ptr<ActivationFunction> Create(const TransformFactoryConfig& config);

    void UpdateActiveOutputCount(std::unique_ptr<BaseConfig> configs[TransformOperationCount],
        uint32_t outputCount) const;

    ActivationFunction(const BaseTransformConfig<ActivationKernel>& config,
        const DataMode& mode, std::unique_ptr<Tensor> pwl);
    ActivationFunction() = delete;
    virtual ~ActivationFunction() = default;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    void ValidateActiveList(ActiveList const & activeList) const override;

    std::unique_ptr<Tensor> Segments;
    std::unique_ptr<PwlCached const> Pwl;

    /** Number of pwl segments constraint - max  */
    static constexpr auto ActivationFunctionSegmentCountMax = uint32_t{ 128 };

protected:
    static std::unique_ptr<PwlCached const> createPwlCached(uint32_t elementSize,
        PwlSegment const * segmentsIn, uint32_t segmentCountIn);

    virtual void updateExecutionKernelConfig(ExecutionKernelConfig<ActivationConfig> & config)
        const override
    {
        setSoftwareScratchPad(config);
    }

    static const FullCapabilitiesMap capabilities;
    static const FullCapabilitiesMap outputCapabilities;
};

}
