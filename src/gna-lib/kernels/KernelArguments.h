/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Address.h"
#include "Macros.h"

#include "gna2-model-impl.h"

#include <array>
#include <cstdint>
#include <cstring>

using GNA::BaseAddress;

using GNA::BiasCompound;
using GNA::BiasRegular;
using GNA::WeightScaleFactor;


/** Number of input groups constraint - max */
const uint32_t XNN_N_GROUP_MAX = 8;

/**
 * Structure will hold aligned deinterleaved feature vectors
 * and PWL activation function auxiliary buffers used for performance improvements
 * One structure per thread in thread pool will be created and managed by kernel dispatcher
 */
struct KernelBuffers
{
    KernelBuffers();
    ~KernelBuffers();

    KernelBuffers(const KernelBuffers& rhs) = delete;

    KernelBuffers(KernelBuffers&& rhs) noexcept
    {
        memcpy_s(this, sizeof(*this), &rhs, sizeof(rhs));

        rhs.d0 = nullptr;
        rhs.d1 = nullptr;
        rhs.d2 = nullptr;
        rhs.d3 = nullptr;
        rhs.d4 = nullptr;
        rhs.d5 = nullptr;
        rhs.d6 = nullptr;
        rhs.d7 = nullptr;
        rhs.pool = nullptr;
        rhs.cnnFusedBuffer = nullptr;
    }

    void ReallocateCnnScratchPad(uint32_t cnnScratchSize);

    int16_t *d0 = nullptr;
    int16_t *d1 = nullptr;
    int16_t *d2 = nullptr;
    int16_t *d3 = nullptr;
    int16_t *d4 = nullptr;
    int16_t *d5 = nullptr;
    int16_t *d6 = nullptr;
    int16_t *d7 = nullptr;
    int64_t *pool = nullptr;
    int8_t *cnnFusedBuffer = nullptr;
    uint32_t cnnFusedBufferSize = 0;
};

namespace GNA
{
struct PwlCached;
}

struct BaseConfig
{
    BaseConfig() = default;
    BaseConfig(const BaseAddress& inputBuffer, const BaseAddress& outputBuffer);

    bool SetBuffer(uint32_t operandIndex, const BaseAddress& buffer)
    {
        if (operandIndex > GNA::ScratchpadOperandKernelIndex)
        {
            return false;
        }
        Buffers[operandIndex] = buffer;
        if (GNA::InputOperandIndex == operandIndex)
        {
            Inputs = buffer;
        }
        else if (GNA::OutputOperandIndex == operandIndex)
        {
            Outputs = buffer;
        }
        return  true;
    }

    int8_t const * Inputs = nullptr;
    int8_t * Outputs = nullptr;
    std::array<int8_t *, GNA::ScratchpadOperandKernelIndex + 1> Buffers;
};

template<typename TransformConfig>
struct KernelConfig : public BaseConfig
{
    using BaseConfig::BaseConfig;
    KernelConfig(KernelConfig const & source) = default;
    KernelConfig(TransformConfig const & source, BaseConfig const & io) :
        BaseConfig{ io },
        Transform{ source }
    {}

    TransformConfig Transform;
};

struct ExecutionConfig
{
    ExecutionConfig(KernelBuffers * intermediate, uint32_t * saturationCount, uint32_t const * bufferElementCount) :
        Intermediate{ intermediate },
        SaturationCount{ saturationCount },
        BufferElementCount{ bufferElementCount }
    {};

    KernelBuffers * const Intermediate;
    uint32_t * const SaturationCount;
    uint32_t const * const BufferElementCount;
};

template<typename TransformConfig>
struct ExecutionKernelConfig : public ExecutionConfig
{
    ExecutionKernelConfig(KernelConfig<TransformConfig> const & requestConfig,
        ExecutionConfig const & executionConfig) :
        ExecutionConfig{ executionConfig },
        RequestConfig{ requestConfig }
    {}

    KernelConfig<TransformConfig> RequestConfig;
};

struct ActivationConfig
{
    ActivationConfig(uint32_t elementCount, GNA::PwlCached const * kernel);

    uint32_t ElementCount;
    GNA::PwlCached const * const Kernel;
};

struct AffineConfig
{
    AffineConfig(int16_t const * inputIn, int32_t * const outputIn, AffineConfig const * const source);
    AffineConfig(AffineConfig const * const source, ExecutionConfig const & executionConfigIn);
    AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn,
        uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn, void const * weightsIn,
        void const * biases, void const * multiBiasIn, uint32_t const multiBiasVectorCountIn);
    AffineConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn,
        uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn, void const * weightsIn,
        void const * biases, void const * multiBiasIn, uint32_t const multiBiasVectorCountIn,
        const uint32_t bytesPerBiasIn);

    uint32_t const outputElementCount;  // M - out rows
    uint32_t const inputVectorCount;    // N - columns
    uint32_t const inputElementCount;   // K - rows
    int16_t const * input;              // I - (interleaved) [K;N]
    int32_t * output;                   // O - [M;N]
    ExecutionConfig const * execution;
    union
    {
        int8_t const * const weights1B;     // W - [M;K]
        int16_t const * const weights2B;    // W - [M;K]
    };
    union
    {
        WeightScaleFactor const * const weightScaleFactors; // [M] Scaling factors for 1B weights or NULL for 2B weights.
        BiasCompound const * const biasesCompound;     // B - [M]
        BiasRegular const * const biasesSimple;       // B - [M]
    };
    void const * const multiBias;
    uint32_t const multiBiasVectorCount;
    uint32_t const bytesPerBias = 0;
};

struct AffineConfigAl
{
    AffineConfigAl(uint32_t const * indicesIn, uint32_t const countIn);

    uint32_t const * const indices; // AL [L]
    uint32_t const count;           // L
};

struct RecurrentConfig
{
    RecurrentConfig(
        uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn,
        int16_t const * inputIn, int16_t * const feedbackBufferIn, int32_t * const outputIn,
        int16_t * outputActivatedIn, void const * weightsIn, void const * biases, ActivationConfig const & pwl);

    RecurrentConfig(
        uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn,

        int16_t const * inputIn, int16_t * const feedbackBufferIn, int32_t * const outputIn,
        int16_t * outputActivatedIn, void const * weightsIn, void const * biases,
        uint32_t bytesPerBiasIn, uint32_t bytesPerOutputIn, ActivationConfig const & pwl);

    uint32_t const outputElementCount;      // M - cols
    uint32_t const inputVectorCount;        // N - rows
    uint32_t const inputElementCount;       // K - cols
    int16_t const * input;                  // I - (flat) [N;K]
    int16_t * feedbackBuffer;               // (flat) [N,M]
    int32_t * output;                       // O1 - [N,M]
    uint32_t bytesPerBias = 0;
    uint32_t bytesPerOutput = 0;
    union
    {
        int8_t const * const weights1B;         // W - [M,K+M]
        int16_t const * const weights2B;        // W - [M,K+M]
    };
    union
    {
        BiasCompound const * const biasesCompound; // B - [M]
        BiasRegular const * const biasesSimple;   // B - [M]
    };
    KernelConfig<ActivationConfig> activation;
};

struct TransposeConfig
{
    static TransposeConfig MakeFrom(
        ExecutionKernelConfig<AffineConfig> const * const config);

    TransposeConfig(uint32_t rowCountIn, uint32_t columntCountIn,
        int16_t const * const inputIn, int16_t * const outputIn);

    uint32_t const rowCount;
    uint32_t const columnCount;
    int16_t const * input;
    int16_t * output;
};

struct CopyConfig
{
    CopyConfig(uint32_t rowCountIn, uint32_t columntCountIn, uint32_t inputColumnCountIn, uint32_t outputColumnCountIn,
        int16_t const * const inputIn, int16_t * const outputIn);

    uint32_t const rowCount;
    uint32_t const columnCount;
    uint32_t const inputColumnCount;
    uint32_t const outputColumnCount;
    int16_t const * input;
    int16_t * output;
};

struct ConvolutionConfig
{
    ConvolutionConfig(ConvolutionConfig const * const source, int16_t const * const inputsIn,
        int32_t * const outputsIn);
    ConvolutionConfig(ConvolutionConfig const * const source, ExecutionConfig const & executionConfigIn);
    ConvolutionConfig(uint32_t const inputBandStrideIn, uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
        uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn, int16_t const * const filtersIn,
        BiasRegular const * const biasesIn, int32_t * const outputsIn);
    ConvolutionConfig(uint32_t const inputBandStrideIn, uint32_t const FilterOutputCountIn, uint32_t const FilterCountIn,
        uint32_t const FilterCoefficientCountIn, int16_t const * const inputsIn, int16_t const * const filtersIn,
        BiasRegular const * const biasesIn, int32_t * const outputsIn, uint32_t bytesPerBiasIn, uint32_t bytesPerFilterIn);

    uint32_t const inputBandStride;
    uint32_t const filterOutputCount;
    uint32_t const filterCount;
    uint32_t const filterCoefficientCount;
    uint32_t const bytesPerBias = 0;
    uint32_t const bytesPerFilter = 0;

    int16_t const * inputs;
    int16_t const * const filters;
    BiasRegular const * const biases;

    union
    {
        int32_t * convolutedOutputs;
        int16_t * pooledOutputs;
    };
    ExecutionConfig const * execution;
};

struct GmmConfig
{
    GmmConfig(uint32_t const inputVectorCountIn, uint32_t const inputElementCountIn, uint32_t const mixCountIn,
        uint32_t const meanSetOffsetSizeIn, uint32_t const varSetOffsetSizeIn, uint32_t const gaussConstSetOffsetSizeIn,
        uint32_t const maxScoreIn, uint32_t const stateCountIn,
        uint8_t const * means, uint8_t const * vars, uint32_t const * gconst);

    uint32_t const InputVectorCount;
    uint32_t const InputElementCount;
    uint32_t const InputElementOffset;
    uint32_t const MixtureCount;
    uint32_t const MeanSetOffsetSize;
    uint32_t const VarSetOffsetSize;
    uint32_t const GaussConstSetOffsetSize;
    uint32_t const MaxScore;
    uint32_t StateCount;

    uint8_t const * Means;
    union
    {
        uint8_t const * Vars;
        uint16_t const * Vars16;
    };
    uint32_t const * Gconst;

    uint8_t const * Input;
    uint32_t * Output;
};
