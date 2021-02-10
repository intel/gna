/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Address.h"
#include "common.h"
#include "DataMode.h"
#include "GnaConfig.h"
#include "GnaTypes.h"
#include "HwModuleInterface.hpp"
#include "LayerDescriptor.h"

#include "gna-api-types-gmm.h"
#include "gna2-common-impl.h"

#include <cstdint>
#include <map>
#include <memory>

namespace GNA
{

class ActivationFunction;
class Layer;
class PoolingFunction2D;
struct AffineFunction;
struct ConvolutionFunction2D;
struct FiltersTensor;

struct DescriptorParameters
{
    DescriptorParameters(Layer const & softwareLayer,
        const LayerDescriptor& xnnDescriptor,
        HwModuleInterface const & hwModule);

    virtual ~DescriptorParameters() = default;

    Layer const & SoftwareLayer;
    LayerDescriptor XnnDescriptor;
    const AddrGmmCfg GmmDescriptor;
    GetHwOffset GetBufferOffset;
    HwModuleInterface const & HwModule;
};

// Hardware Layer descriptor converter
class HardwareLayer : public DescriptorParameters
{
public:
    static std::unique_ptr<HardwareLayer> Create(const DescriptorParameters& parameters);
    virtual ~HardwareLayer() = default;

    virtual NN_OP_TYPE GetNnopType(bool hasActiveList) const;

    virtual uint32_t GetXnnDescriptorOffset() const;
    virtual uint32_t GetGmmDescriptorOffset() const;

    virtual uint32_t GetLdNnopOffset() const;

    virtual uint32_t GetLdScrlenOffset() const;
    virtual uint32_t GetLdGmmMeanOffset() const;
    virtual uint32_t GetLdGmmInverseCovarianceOffset() const;
    virtual uint32_t GetLdGaussianConstantOffset() const;

    virtual uint32_t GetLdInputOffset() const;
    virtual uint32_t GetLdOutputOffset() const;

    virtual uint32_t GetLdWeightOffset() const;
    virtual uint32_t GetLdBiasOffset() const;
    virtual uint32_t GetLdFilterOffset() const;
    virtual uint32_t GetLdIntermediateOutputOffset() const;
    virtual uint32_t GetLdWeightScaleFactorOffset() const;
    virtual uint32_t GetLdPwlOffset() const;

    virtual uint32_t GetLdFeedbackOffset() const;

    virtual uint32_t GetLdActlenOffset() const;
    virtual uint32_t GetLdActlistOffset() const;

    virtual uint32_t GetScrlen(uint32_t indicesCount) const;

protected:
    static const std::map<const nn_operation, const NN_OP_TYPE> OperationsMap;

    HardwareLayer(const DescriptorParameters& parameters);

    void saveCommonPart();
    void save();
    void saveActivation(const ActivationFunction* activationIn);
};

// Extended Hardware Layer descriptor converter
class HardwareLayerExt : public HardwareLayer
{
public:
    HardwareLayerExt(const HardwareLayerExt &) = delete;
    HardwareLayerExt& operator=(const HardwareLayerExt&) = delete;
    virtual ~HardwareLayerExt() = default;

protected:
    HardwareLayerExt(const DescriptorParameters& parameters, uint32_t iterationGrouping);
    HardwareLayerExt(const DescriptorParameters& parameters);

    static uint32_t calculateEffectiveInputSizeFor3_0(const DescriptorParameters& parameters);
    void save();

    const uint32_t bufferElementCount;
    uint32_t lastIterationElementCount;
    const AffineFunction* affine = nullptr;

private:
    uint32_t iterationCount; // number of iterations = data chunks/parts
};

// Affine, Diagonal and transpose layers Layer descriptor converter
class HardwareLayerAffDiagTrans : public HardwareLayerExt
{
public:
    HardwareLayerAffDiagTrans(const DescriptorParameters& parameters);
    virtual ~HardwareLayerAffDiagTrans() = default;

    virtual NN_OP_TYPE GetNnopType(bool hasActiveList) const override;
};

class HardwareLayerAffineMBias : public HardwareLayerExt
{
public:
    HardwareLayerAffineMBias(const DescriptorParameters& parameters);
    virtual ~HardwareLayerAffineMBias() = default;
};

// Hardware Copy Layer descriptor converter
class HardwareLayerCopy : public HardwareLayer
{
public:
    HardwareLayerCopy(const HardwareLayerCopy &) = delete;
    HardwareLayerCopy& operator=(const HardwareLayerCopy&) = delete;
    HardwareLayerCopy(const DescriptorParameters& parameters);
    virtual ~HardwareLayerCopy() = default;

protected:
    void save();
};

// Recurrent Layer descriptor converter
class HardwareLayerRnn : public HardwareLayerExt
{
public:
    HardwareLayerRnn(const HardwareLayerRnn &) = delete;
    HardwareLayerRnn& operator=(const HardwareLayerRnn&) = delete;
    HardwareLayerRnn(const DescriptorParameters& parameters);
    virtual ~HardwareLayerRnn() = default;

    virtual uint32_t GetLdFeedbackOffset() const override;

protected:
    void convert();
    void save();

private:
    uint32_t feedbackIterationsCount;
    uint32_t feedbackFirstIterElementCount; // number of el. in first feedback data iter.
    uint32_t feedbackLastIterElementCount; // number of el. in last feedback data iter.
};

// Convolutional Layer descriptor converter
class HardwareLayerCnn : public HardwareLayerExt
{
public:
    HardwareLayerCnn(const HardwareLayerRnn &) = delete;
    HardwareLayerCnn& operator=(const HardwareLayerRnn&) = delete;
    HardwareLayerCnn(const DescriptorParameters& parameters);
    virtual ~HardwareLayerCnn() = default;

protected:
    void save();

private:
    static const uint32_t CNN_N_FLT_ITER_MAX = 16; // CNN maximum number of filters per iteration

    uint32_t filtersIterationCount;                // Number of iterations  to process all filters.
    uint32_t filtersCountInLastIteration;          // Number of filters in last iteration.
    uint32_t filtersCountInFullIteration;          // Number of filters in buffer in full iterations.
    uint32_t filtersElementCountInFullIteration;   // Size of filter in non-last iterations (elements).
    uint32_t filtersElementCountInLastIteration;   // Size of filter in last iterations (elements).
    uint32_t outputElementCount;                   // Number of final output elements
    uint32_t convOutputElementCount;               // Number of output elements after convolution and before downsampling
};

// Convolutional Layer descriptor converter
class HardwareLayerCnn2D : public HardwareLayer
{
public:
    HardwareLayerCnn2D(const HardwareLayerRnn &) = delete;
    HardwareLayerCnn2D& operator=(const HardwareLayerRnn&) = delete;
    HardwareLayerCnn2D(const DescriptorParameters& parameters);
    virtual ~HardwareLayerCnn2D() = default;

    static uint32_t GetKernelMemorySize(DeviceVersion deviceVersion,
        FiltersTensor const * filter);

    static uint32_t GetConvolutionMemorySize(DeviceVersion deviceVersion,
        ConvolutionFunction2D const * cnnIn);

    static uint32_t GetPoolingMemorySize(DeviceVersion deviceVersion,
        PoolingFunction2D const * poolingIn, const DataMode& outputMode);

protected:
    void save();

    void save1D();

    HwUarchParams CalculateUArchConfig() const;

    ConvolutionFunction2D const * const cnn;
    PoolingFunction2D const * const pooling;
    bool const is1D = false;

private:
    static const uint32_t CNN_N_FLT_ITER_MAX = 16; // CNN maximum number of filters per iteration

    HwUarchParams uArchConfig;
};

// Hardware GMM Layer descriptor converter
class HardwareLayerGmm : public HardwareLayer
{
public:
    HardwareLayerGmm(const HardwareLayerGmm &) = delete;
    HardwareLayerGmm& operator=(const HardwareLayerGmm&) = delete;
    HardwareLayerGmm(const DescriptorParameters& parameters);
    virtual ~HardwareLayerGmm() = default;

    virtual NN_OP_TYPE GetNnopType(bool hasActiveList) const override;
    virtual uint32_t GetLdOutputOffset() const override;
    virtual uint32_t GetLdInputOffset() const override;
    virtual uint32_t GetGmmDescriptorOffset() const override;
    virtual uint32_t GetLdActlistOffset() const override;
    virtual uint32_t GetLdActlenOffset() const override;
    virtual uint32_t GetLdScrlenOffset() const override;
    virtual uint32_t GetScrlen(uint32_t indicesCount) const override;

protected:
    static const std::map<const gna_gmm_mode, const GMM_MODE_CTRL> GmmModes;

    void save();
};
}
