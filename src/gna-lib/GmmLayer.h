/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Bias.h"
#include "KernelArguments.h"
#include "Layer.h"
#include "Weight.h"

#include "common.h"
#include "gmm.h"
#include "gna-api-types-gmm.h"

#include <cstdint>
#include <map>

namespace GNA
{
class BaseValidator;
struct ActiveList;
struct LayerConfiguration;

// GMM Calculation configuration
class GmmOperation : public Layer
{
public:
    template<class T>
    GmmOperation(const T& layer, const BaseValidator& validatorIn) :
        Layer(layer, validatorIn, { GmmTransform }, BaseAddress())
    {}

    virtual ~GmmOperation() = default;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    virtual void VerifyHas1BInputAnd2BWeight() override;

    virtual DataConfig GetDataMode() const override;
};

class GmmFunction : public TransformAl<GmmConfig, GmmMaxMix, GmmMaxMixActiveList>
{
public:
    static std::unique_ptr<GmmFunction> Create(
        const TransformFactoryConfig& config,
        const OperationConfig& operation);

    virtual ~GmmFunction() = default;

    Tensor const& GetOperand(uint32_t operandIndex) const override;

    void ValidateActiveList(ActiveList const & activeList) const override;

    virtual DataConfig GetDataMode() const = 0;

    std::unique_ptr<const WeightTensor> Means;
    std::unique_ptr<const WeightTensor> InverseCovariances;
    std::unique_ptr<const BiasTensor> GaussianConstants;

    BaseAddress MeanBuffer;
    BaseAddress InverseCovarianceBuffer;
    BaseAddress GaussianConstantBuffer;

    uint32_t InverseCovarianceSize;
    uint32_t const MaximumScore;
    uint32_t MeanSetOffsetSize;
    uint32_t VarSetOffsetSize;
    uint32_t GaussConstSetOffsetSize;
    uint32_t StateCount;

protected:
    GmmFunction(const BaseTransformConfig<GmmMaxMix>& config,
        std::unique_ptr<const WeightTensor> means,
        std::unique_ptr<const WeightTensor> inverseCovariances,
        std::unique_ptr<const BiasTensor> gaussianConstants,
        uint32_t const maximumScore,
        const KernelMap<GmmMaxMixActiveList>& kernelsAl);

    void InitHiddenConfig();

    static const FullCapabilitiesMap & getOutputCapabilities();
};

class GmmFunctionFlat : public GmmFunction
{
public:
    GmmFunctionFlat(const BaseTransformConfig<GmmMaxMix>& config,
        std::unique_ptr<const WeightTensor> means,
        std::unique_ptr<const WeightTensor> inverseCovariances,
        std::unique_ptr<const BiasTensor> gaussianConstants,
        uint32_t const maximumScore,
        const KernelMap<GmmMaxMixActiveList>& kernelsAl);

    virtual ~GmmFunctionFlat() = default;

    virtual DataConfig GetDataMode() const override;
};

class GmmFunctionInterleaved : public GmmFunction
{
public:
    GmmFunctionInterleaved(const BaseTransformConfig<GmmMaxMix>& config,
        std::unique_ptr<const WeightTensor> interleavedData,
        uint32_t const maximumScore,
        const KernelMap<GmmMaxMixActiveList>& kernelsAl);

    virtual ~GmmFunctionInterleaved() = default;

    virtual DataConfig GetDataMode() const override;
};


}
