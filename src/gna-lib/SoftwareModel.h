/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "HardwareRequest.h"
#include "IScorable.h"
#include "Layer.h"
#include "Logger.h"
#include "ModelError.h"


#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{

class BaseValidator;
class RequestConfiguration;
class RequestProfiler;

class SoftwareModel : public IScorable
{
public:
    static void LogAcceleration(AccelerationMode accel)
    {
        auto name = accel.GetName();
        Log->Message("Processing using %s acceleration\n", name);
    }
    static void LogOperationMode(GnaOperationMode mode)
    {
        if (mode == GMM)
        {
            Log->Message("Processing using GMM operation mode\n");
        }
        else if (mode == xNN)
        {
            Log->Message("Processing using xNN operation mode\n");
        }
    }

    SoftwareModel(const Gna2Model& model,
        BaseValidator const && softwareOnlyValidator,
        const std::vector<Gna2AccelerationMode>& supportedCpuAccelerationsIn);

    SoftwareModel(const Gna2Model& model,
        BaseValidator const && softwareOnlyValidator,
        BaseValidator const && hwConsistentValidator,
        const std::vector<Gna2AccelerationMode>& supportedCpuAccelerationsIn,
        const std::vector<std::unique_ptr<SubModel>>& subModels);

    SoftwareModel(const SoftwareModel &) = delete;
    SoftwareModel& operator=(const SoftwareModel&) = delete;
    virtual ~SoftwareModel() = default;

    void Score(ScoreContext & context) override;

    uint32_t GetMaximumOperandSize(uint32_t operandIndex);

    Layer const& GetLayer(uint32_t layerIndex) const;

    std::vector<std::unique_ptr<Layer>> const& GetLayers() const
    {
        return layers;
    }

    auto const & GetBufferConfigValidator() const
    {
        return bufferConfigValidator;
    }

private:
    SoftwareModel(const Gna2Model& model,
        const std::vector<Gna2AccelerationMode>& supportedCpuAccelerationsIn);

    void build(const Gna2Operation* operations,
        const BaseValidator & softwareOnlyValidator,
        const BaseValidator & hwConsistentValidator,
        const std::vector<std::unique_ptr<SubModel>>& subModels);

    void buildSingleLayer(std::unique_ptr<Layer> & layer);

    uint32_t FindMaximumOperandSize(uint32_t operandIndex) const;

    static void FindMaximumOperandSizeForSingleLayer(Layer const & layer, uint32_t operandIndex,
        uint32_t & maxSize);

    std::vector<std::unique_ptr<Layer>> layers;

    uint32_t const layerCount;

    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerations;

    std::map<uint32_t /* operandIndex */, uint32_t> maximumOperandSizes;

    BufferConfigValidator bufferConfigValidator;
};

struct InferenceConfig
{
    typedef ExecutionConfig& (InferenceConfig::*GetEffectiveMethod)(Layer const & layer) const;

    InferenceConfig(KernelBuffers *fvBuffers, RequestConfiguration const &requestConfiguration);

    ExecutionConfig& GetEffective(Layer& layer) const
    {
        return (this->*getEffective)(layer);
    }

    // scoring saturation counter
    uint32_t SaturationCount;

private:
    GetEffectiveMethod getEffective;

    ExecutionConfig& getNormal(Layer const & layer) const;
    ExecutionConfig& getFor3_0Fix(Layer const & layer) const;

    // if ADL consistency is active
    bool has3_0Consistency = false;

    // config for usual inference request
    std::unique_ptr<ExecutionConfig> executionConfig;

    // config for inference request for ADL
    std::unique_ptr<ExecutionConfig> executionConfig3_0;
};

}
