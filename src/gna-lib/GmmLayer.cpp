/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "GmmLayer.h"

#include "AccelerationDetector.h"
#include "Address.h"
#include "ActiveList.h"
#include "Capabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <algorithm>
#include <memory>

using namespace GNA;

void GmmOperation::VerifyHas1BInputAnd2BWeight()
{}

Tensor const & GmmOperation::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case GmmMeanOperandIndex://[[fallthrough]] also same value for GmmInterleavedOperandIndex
    case GmmInverseCovarianceOperandIndex://[[fallthrough]]
    case GmmGaussianConstantOperandIndex:
    {
        return getTransformOperand(GmmTransform, operandIndex);
    }
    default:
        return Layer::GetOperand(operandIndex);
    }
}

DataConfig GmmOperation::GetDataMode() const
{
    return reinterpret_cast<GmmFunction const *>(inputTransform)->GetDataMode();
}

std::unique_ptr<GmmFunction> GmmFunction::Create(const TransformFactoryConfig& config,
    const OperationConfig& operation)
{
    auto const isFlat = operation.Operation->NumberOfOperands != 3;
    auto varMode = GNA_UINT8;
    std::unique_ptr<const WeightTensor> means;
    std::unique_ptr<const WeightTensor> inverseCovariances;
    std::unique_ptr<const BiasTensor> gaussianConstants;

    Shape expectedMeans = Shape{ GNA_TENSOR_HWD, config.output->Dimensions.at('H'), 1u, config.input->Dimensions.at('W') };

    std::function<void()> command = [&]()   // Common for flat and interleaved
    {
        auto const meanTensor = operation.GetEnabledOperand(GmmMeanOperandIndex);
        expectedMeans['W'] = meanTensor.Shape.Dimensions[1];
        expectedMeans.ExpectEqualInverted(meanTensor.Shape);
        means = std::make_unique<const WeightTensor>(meanTensor, config.validator);
    };
    static_assert(GmmMeanOperandIndex == GmmInterleavedOperandIndex, "");
    ModelErrorHelper::ExecuteForModelItem(command, GmmMeanOperandIndex);

    if (isFlat)
    {
        const auto expectedGaussianConstants = Shape{ GNA_TENSOR_HW,
            config.output->Dimensions.at('H'),
            Gna2RoundUp(expectedMeans.at('W'), 2) };

        command = [&]()
        {
            auto const inverseCovariancesTensor = operation.GetEnabledOperand(GmmInverseCovarianceOperandIndex);
            expectedMeans.ExpectEqualInverted(inverseCovariancesTensor.Shape);
            inverseCovariances = std::make_unique<const WeightTensor>(inverseCovariancesTensor, config.validator);
        };
        ModelErrorHelper::ExecuteForModelItem(command, GmmInverseCovarianceOperandIndex);
        command = [&]()
        {
            auto const gaussianConstantsTensor = operation.GetEnabledOperand(GmmGaussianConstantOperandIndex);
            expectedGaussianConstants.ExpectEqualInverted(gaussianConstantsTensor.Shape);
            gaussianConstants = std::make_unique<const BiasTensor>(
                gaussianConstantsTensor, 0, Gna2BiasModeDefault, config.validator);
        };
        ModelErrorHelper::ExecuteForModelItem(command, GmmGaussianConstantOperandIndex);

        varMode = inverseCovariances->Mode.Value;
    }
    else
    {
        varMode = means->Mode.Value;
    }

    auto const kernelMode = KernelMode{ GNA_UINT8, varMode, GNA_UINT32 };
    const auto& gmmKernels = AccelerationDetector::GetKernelMap<GmmMaxMix>(KERNEL_GMM, kernelMode);
    const auto& gmmKernelsAl = AccelerationDetector::GetKernelMap<GmmMaxMixActiveList>(KERNEL_GMM_AL, kernelMode);
    auto const maximumScore = operation.GetParameterAs<uint32_t>(0);

    if (isFlat)
    {
        return std::make_unique<GmmFunctionFlat>(
            BaseTransformConfig<GmmMaxMix>{config, gmmKernels},
            std::move(means), std::move(inverseCovariances), std::move(gaussianConstants),
            maximumScore, gmmKernelsAl);
    }
    return std::make_unique<GmmFunctionInterleaved>(
        BaseTransformConfig<GmmMaxMix>{config, gmmKernels},
        std::move(means),
        maximumScore, gmmKernelsAl);
}

Tensor const& GmmFunction::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case GmmMeanOperandIndex:
    {
        return GetOperandIfExistOrThrow(Means);
    }
    case GmmInverseCovarianceOperandIndex:
    {
        return GetOperandIfExistOrThrow(InverseCovariances);
    }
    case GmmGaussianConstantOperandIndex:
    {
        return GetOperandIfExistOrThrow(GaussianConstants);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}

void GmmFunction::ValidateActiveList(ActiveList const & activeList) const
{
    Expect::InRange(activeList.IndicesCount,
        ui32_1, Means->at(GNA_DIM_H), Gna2StatusActiveListIndicesInvalid);
}

GmmFunction::GmmFunction(const BaseTransformConfig<GmmMaxMix>& config,
    std::unique_ptr<const WeightTensor> means,
    std::unique_ptr<const WeightTensor> inverseCovariances,
    std::unique_ptr<const BiasTensor> gaussianConstants,
    uint32_t const maximumScore,
    const KernelMap<GmmMaxMixActiveList>& gmmKernelsAl) :
    TransformAl{ GmmTransform, &config.kernels, &gmmKernelsAl, config.input },
    Means{ std::move(means) },
    InverseCovariances{ std::move(inverseCovariances) },
    GaussianConstants{ std::move(gaussianConstants) },
    MaximumScore{ maximumScore }
{
    Expect::NotNull(Means);

    MeanBuffer = Means->Buffer;
    StateCount = Means->at(GNA_DIM_H);

    auto const mixCount = Means->at(GNA_DIM_W);
    auto const inElementCount = Input->at(GNA_DIM_W);
    InverseCovarianceSize = InverseCovariances ? InverseCovariances->Mode.Size : Means->Mode.Size;

    MeanSetOffsetSize = mixCount * inElementCount * GMM_MEAN_VALUE_SIZE;
    VarSetOffsetSize = mixCount * inElementCount * InverseCovarianceSize;
    GaussConstSetOffsetSize = RoundUp(mixCount, 2) * GMM_CONSTANTS_SIZE;

    Expect::InRange(MeanSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
        GMM_MIXTURE_COMP_COUNT_MAX * GMM_FV_ELEMENT_COUNT_MAX * GMM_MEAN_VALUE_SIZE, Gna2StatusGmmBadMeanSetoff);
    Expect::MultiplicityOf(MeanSetOffsetSize, GMM_MEM_ALIGNMENT);
    Expect::InRange(VarSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
        GMM_MIXTURE_COMP_COUNT_MAX * GMM_FV_ELEMENT_COUNT_MAX * GMM_COVARIANCE_SIZE_MAX, Gna2StatusGmmBadVarSetoff);
    Expect::MultiplicityOf(VarSetOffsetSize, GMM_MEM_ALIGNMENT);
    Expect::InRange(GaussConstSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
        GMM_MIXTURE_COMP_COUNT_MAX * GMM_CONSTANTS_SIZE, Gna2StatusGmmBadGconstOffset);
    Expect::MultiplicityOf(GaussConstSetOffsetSize, GMM_MEM_ALIGNMENT);

    Output = std::make_unique<Tensor>(config.output->Dimensions, config.output->Mode,
        config.output->Buffer, Validator{ config.validator, getOutputCapabilities() });
}

void GmmFunction::InitHiddenConfig()
{
    auto const gmmConfig = GmmConfig{ Input->at(GNA_DIM_H),
       Input->at(GNA_DIM_W),
       Means->at(GNA_DIM_W),
       MeanSetOffsetSize,
       VarSetOffsetSize,
       GaussConstSetOffsetSize,
       MaximumScore,
       StateCount,
       MeanBuffer,
       InverseCovarianceBuffer,
       GaussianConstantBuffer,
    };
    hiddenConfig = std::make_unique<KernelConfig<GmmConfig>>(gmmConfig,
        BaseConfig{ Input->Buffer, Output->Buffer });
}

const FullCapabilitiesMap& GmmFunction::getOutputCapabilities()
{
    static const FullCapabilitiesMap capabilities =
    {
     {INTEL_GMM, {
        {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW}, // H - GMM States, W - grouping
            {{GNA_DIM_W, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
            { { GNA_UINT32, GNA_DATA_ACTIVATION_DISABLED }, Gna2StatusXnnErrorOutputBytes }})}
    }},
    };
    return capabilities;
}

GmmFunctionFlat::GmmFunctionFlat(
    const BaseTransformConfig<void(*)(ExecutionKernelConfig<GmmConfig> const* const)>& config,
    std::unique_ptr<const WeightTensor> means, std::unique_ptr<const WeightTensor> inverseCovariances,
    std::unique_ptr<const BiasTensor> gaussianConstants, uint32_t const maximumScore,
    const KernelMap<void(*)(ExecutionKernelConfig<GmmConfig> const* const, AffineConfigAl al)>&kernelsAlIn) :
    GmmFunction{ config, std::move(means), std::move(inverseCovariances), std::move(gaussianConstants), maximumScore, kernelsAlIn }
{
    Expect::NotNull(GaussianConstants);
    Expect::NotNull(InverseCovariances);
    Expect::Equal(Means->Mode.Type, Gna2DataTypeUint8, Gna2StatusDataModeInvalid);

    InverseCovarianceBuffer = InverseCovariances->Buffer;
    GaussianConstantBuffer = GaussianConstants->Buffer;

    config.validator.ValidateBufferIfSet(GaussianConstants->Buffer, StateCount * GaussConstSetOffsetSize, {8, Gna2StatusMemoryAlignmentInvalid });
    config.validator.ValidateBufferIfSet(Means->Buffer, StateCount * MeanSetOffsetSize, { 8, Gna2StatusMemoryAlignmentInvalid });
    config.validator.ValidateBufferIfSet(InverseCovariances->Buffer, StateCount * VarSetOffsetSize, { 8, Gna2StatusMemoryAlignmentInvalid });

    InitHiddenConfig();
}

DataConfig GmmFunctionFlat::GetDataMode() const
{
    return DataConfig(Input->Mode, InverseCovariances->Mode, GNA_UINT32, Output->Mode);
}

GmmFunctionInterleaved::GmmFunctionInterleaved(
    const BaseTransformConfig<void(*)(ExecutionKernelConfig<GmmConfig> const* const)>& config,
    std::unique_ptr<const WeightTensor> interleavedData, uint32_t const maximumScore,
    const KernelMap<GmmMaxMixActiveList>& kernelsAlIn) :
    GmmFunction{ config, std::move(interleavedData), nullptr, nullptr,
        maximumScore, kernelsAlIn }
{
    InverseCovarianceBuffer = Means->Buffer + MeanSetOffsetSize;
    GaussianConstantBuffer = InverseCovarianceBuffer + VarSetOffsetSize;

    auto const interleavedSetOffsetSize = MeanSetOffsetSize + VarSetOffsetSize + GaussConstSetOffsetSize;;
    MeanSetOffsetSize = interleavedSetOffsetSize;
    VarSetOffsetSize = interleavedSetOffsetSize;
    GaussConstSetOffsetSize = interleavedSetOffsetSize;

    config.validator.ValidateBufferIfSet(Means->Buffer, StateCount * MeanSetOffsetSize);

    InitHiddenConfig();
}

DataConfig GmmFunctionInterleaved::GetDataMode() const
{
    return DataConfig(Input->Mode, Means->Mode, GNA_UINT32, Output->Mode);
}
