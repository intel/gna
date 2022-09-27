/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Address.h"
#include "KernelArguments.h"

#include <map>
#include <utility>

BaseConfig::BaseConfig(const BaseAddress& inputBuffer, const BaseAddress& outputBuffer) :
    Inputs{ inputBuffer },
    Outputs{ outputBuffer },
    Buffers{ inputBuffer, outputBuffer }
{
}

ActivationConfig::ActivationConfig(uint32_t elementCount, GNA::PwlCached const * kernel) :
    ElementCount{elementCount},
    Kernel{kernel}
{}

AffineConfig::AffineConfig(int16_t const * inputIn, int32_t * const outputIn,
    AffineConfig const * const source) :
    AffineConfig{*source}
{
    input = inputIn;
    output = outputIn;
}

AffineConfig::AffineConfig(AffineConfig const * const source, ExecutionConfig const & executionConfigIn) :
    AffineConfig{*source}
{
    execution = &executionConfigIn;
}

AffineConfig::AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn,
    uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn,
    void const * weightsIn, void const * biases, void const * multiBiasIn,
    uint32_t const multiBiasVectorCountIn) :
    outputElementCount{outputElementCountIn},
    inputVectorCount{inputVectorCountIn},
    inputElementCount{inputElementCountIn},
    input{inputIn},
    output{outputIn},
    execution{nullptr},
    weights1B{static_cast<int8_t const *>(weightsIn)},
    biasesCompound{static_cast<BiasCompound const *>(biases)},
    multiBias{multiBiasIn},
    multiBiasVectorCount{multiBiasVectorCountIn}
{}

AffineConfig::AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn,
    uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn,
    void const * weightsIn, void const * biases, void const * multiBiasIn,
    uint32_t const multiBiasVectorCountIn,
    const uint32_t bytesPerBiasIn) :
    outputElementCount{outputElementCountIn},
    inputVectorCount{inputVectorCountIn},
    inputElementCount{inputElementCountIn},
    input{inputIn},
    output{outputIn},
    execution{nullptr},
    weights1B{static_cast<int8_t const *>(weightsIn)},
    biasesCompound{static_cast<BiasCompound const *>(biases)},
    multiBias{multiBiasIn},
    multiBiasVectorCount{multiBiasVectorCountIn},
    bytesPerBias{bytesPerBiasIn}
{}

AffineConfigAl::AffineConfigAl(uint32_t const * indicesIn, uint32_t const countIn) :
    indices{indicesIn},
    count{countIn}
{}

RecurrentConfig::RecurrentConfig(uint32_t const outputElementCountIn,
    uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn, int16_t const * inputIn,
    int16_t * const feedbackBufferIn, int32_t * const outputIn, int16_t * outputActivatedIn,
    void const * weightsIn, void const * biases, ActivationConfig const & pwl) :
    outputElementCount{outputElementCountIn},
    inputVectorCount{inputVectorCountIn},
    inputElementCount{inputElementCountIn},
    input{inputIn},
    feedbackBuffer{feedbackBufferIn},
    output{outputIn},
    weights1B{static_cast<int8_t const *>(weightsIn)},
    biasesCompound{static_cast<BiasCompound const *>(biases)},
    activation{pwl, BaseConfig(outputIn, outputActivatedIn)}
{
}

RecurrentConfig::RecurrentConfig(uint32_t const outputElementCountIn,
    uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn, int16_t const * inputIn,
    int16_t * const feedbackBufferIn, int32_t * const outputIn, int16_t * outputActivatedIn,
    void const * weightsIn, void const * biases, uint32_t bytesPerBiasIn, uint32_t bytesPerOutputIn,
    ActivationConfig const & pwl) :
    outputElementCount{outputElementCountIn},
    inputVectorCount{inputVectorCountIn},
    inputElementCount{inputElementCountIn},
    input{inputIn},
    feedbackBuffer{feedbackBufferIn},
    output{outputIn},
    bytesPerBias{bytesPerBiasIn},
    bytesPerOutput{bytesPerOutputIn},
    weights1B{static_cast<int8_t const *>(weightsIn)},
    biasesCompound{static_cast<BiasCompound const *>(biases)},
    activation{pwl, BaseConfig(outputIn, outputActivatedIn)}
{}

TransposeConfig::TransposeConfig(uint32_t rowCountIn, uint32_t columntCountIn,
    int16_t const * const inputIn, int16_t * const outputIn) :
    rowCount{rowCountIn},
    columnCount{columntCountIn},
    input{inputIn},
    output{outputIn}
{}

TransposeConfig TransposeConfig::MakeFrom(
        ExecutionKernelConfig<AffineConfig> const * const config)
{
    return TransposeConfig {
        config->RequestConfig.Transform.inputElementCount,
        config->RequestConfig.Transform.inputVectorCount,
        reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs),
        config->Intermediate->d0 };
}

CopyConfig::CopyConfig(uint32_t rowCountIn, uint32_t columntCountIn, uint32_t inputColumnCountIn,
    uint32_t outputColumnCountIn, int16_t const * const inputIn, int16_t * const outputIn) :
    rowCount{rowCountIn},
    columnCount{columntCountIn},
    inputColumnCount{inputColumnCountIn},
    outputColumnCount{outputColumnCountIn},
    input{inputIn},
    output{outputIn}
{}

ConvolutionConfig::ConvolutionConfig(ConvolutionConfig const * const source,
    int16_t const * const inputsIn, int32_t * const outputsIn) :
    ConvolutionConfig{*source}
{
    inputs = inputsIn;
    convolutedOutputs = outputsIn;
}

ConvolutionConfig::ConvolutionConfig(ConvolutionConfig const * const source,
    ExecutionConfig const & executionConfigIn) :
    ConvolutionConfig{*source}
{
    execution = &executionConfigIn;
}

ConvolutionConfig::ConvolutionConfig(uint32_t const inputBandStrideIn,
    uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
    uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn,
    int16_t const * const filtersIn, BiasRegular const * const biasesIn, int32_t * const outputsIn) :
    inputBandStride{inputBandStrideIn},
    filterOutputCount{FilterOutputCountIn},
    filterCount{FilterCountIn},
    filterCoefficientCount{FilterCoefficientCountIn},
    inputs{inputsIn},
    filters{filtersIn},
    biases{biasesIn},
    convolutedOutputs{outputsIn},
    execution{nullptr}
{}

ConvolutionConfig::ConvolutionConfig(uint32_t const inputBandStrideIn,
    uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
    uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn,
    int16_t const * const filtersIn, BiasRegular const * const biasesIn,
    int32_t * const outputsIn, uint32_t bytesPerBiasIn, uint32_t bytesPerFilterIn) :
    inputBandStride{inputBandStrideIn},
    filterOutputCount{FilterOutputCountIn},
    filterCount{FilterCountIn},
    filterCoefficientCount{FilterCoefficientCountIn},
    bytesPerBias{bytesPerBiasIn},
    bytesPerFilter{bytesPerFilterIn},
    inputs{inputsIn},
    filters{filtersIn},
    biases{biasesIn},
    convolutedOutputs{outputsIn},
    execution{nullptr}
{}

GmmConfig::GmmConfig(uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn,
    uint32_t const mixCountIn, uint32_t const meanSetOffsetSizeIn,
    uint32_t const varSetOffsetSizeIn, uint32_t const gaussConstSetOffsetSizeIn,
    uint32_t const maxScoreIn, uint32_t const stateCountIn,
    uint8_t const * means, uint8_t const * vars, uint32_t const * gconst) :
    InputVectorCount{ inputVectorCountIn },
    InputElementCount{ inputElementCountIn },
    InputElementOffset(Gna2RoundUpTo64(InputElementCount)),
    MixtureCount{ mixCountIn },
    MeanSetOffsetSize{ meanSetOffsetSizeIn },
    VarSetOffsetSize{ varSetOffsetSizeIn },
    GaussConstSetOffsetSize{ gaussConstSetOffsetSizeIn },
    MaxScore{ maxScoreIn },
    StateCount{ stateCountIn },
    Means{ means },
    Vars{ vars },
    Gconst{ gconst },
    Input{ nullptr },
    Output{ nullptr }
{}
