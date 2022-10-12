/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Capabilities.h"
#include "DataMode.h"
#include "Tensor.h"
#include "Macros.h"

#include <map>

namespace GNA
{

using ComponentFullCapabilityMap = std::map<const uint32_t, FullCapabilitiesMap>;

struct LayerCapabilities
{
    static const OperationCapabilityMap& GetOperands(nn_operation operation, uint32_t operandIndex);

    /** Total number of input elements constraint - must be multiple of */
    static constexpr uint32_t InputElementCountMultiplier = 8;

    /** Number of input groups constraint for Copy layer 3.0- max */
    static constexpr uint32_t CopyRowsMax = 255;

    /** Total number of output elements constraint - must be multiple of */
    static constexpr uint32_t RecurrentOutputElementCountMultiplier = 32;

    /** Total number of input elements constraint - max elements */
    static constexpr uint32_t InputElementCountMax = UINT16_MAX;

    /** Weight elements size constraint - max size B */
    static constexpr uint32_t WeightElementSizeMax = 2;

    static constexpr auto Input = StaticCaps{ 1u, InputElementCountMax, 1u };

    static constexpr auto InputGroupMax = StaticCaps{ 1u, BatchSizeMax, 1u };

    static constexpr RangeLimits<uint32_t> LegacyInputs = RangeLimits<uint32_t>
    {
        InputElementCountMultiplier,
            InputElementCountMax,
            MultiplierMap{ 0, 0, 0,
               2 * InputElementCountMultiplier,
               1 * InputElementCountMultiplier,
               InputElementCountMultiplier / 2
        },
        Gna2StatusXnnErrorInputVolume
    };


    static constexpr auto InputEqual1 = StaticCaps{ 1u, 1u, 1u };
    static constexpr auto Input1D = StaticCaps{ 1u, InputElementCountMax, 8u };
    static constexpr auto WeightMultiplier = StaticCaps{ InputElementCountMultiplier, InputElementCountMax, InputElementCountMultiplier };
    static constexpr auto OutputRnn = StaticCaps{ RecurrentOutputElementCountMultiplier, InputElementCountMax, RecurrentOutputElementCountMultiplier };

    static const DataModeLimits & GetCommonModes(uint32_t operandIndex, Gna2DeviceGeneration generation);

    static auto MakeDataModesCartesian(std::vector<Gna2DataType> types,
        std::vector<Gna2TensorMode> modes = { Gna2TensorModeDefault, Gna2TensorModeExternalBuffer })
    {
        auto cartesian = std::vector<DataMode>(types.size() * modes.size());
        for (auto && type : types)
        {
            for (auto && mode : modes)
            {
                cartesian.emplace_back(DataMode{ type, mode });
            }
        }
        return cartesian;
    }

    template<uint32_t min, uint32_t max, uint32_t multipliers, uint32_t operand>
    static auto MakeLimits()
    {
        static const auto limits = RangeLimits<>
        {
            min, max, multipliers, GetError<operand>().first
        };
        return limits;
    }

    template<const std::array<uint32_t, 3>& dimensions, uint32_t operand = InputOperandIndex>
    static auto MakeLimits()
    {
        return MakeLimits<dimensions.at(0), dimensions.at(1), dimensions.at(2), operand>();
    }

    template<const RangeLimits<uint32_t>& limits, uint32_t operand = InputOperandIndex>
    static auto MakeLimitsMulti()
    {
        return RangeLimits<>{limits, GetError<operand>().first};
    }

    template<uint32_t operand>
    static constexpr auto GetError()
    {
        constexpr std::pair<Gna2Status, Gna2Status> errors[] =
        {
            { Gna2StatusXnnErrorInputVolume, Gna2StatusXnnErrorInputBytes }, // InputOperandIndex
            { Gna2StatusXnnErrorOutputVolume, Gna2StatusXnnErrorOutputBytes }, // OutputOperandIndex
            { Gna2StatusXnnErrorWeightVolume, Gna2StatusXnnErrorWeightBytes }, // FilterOperandIndex
            { Gna2StatusXnnErrorBiasVolume, Gna2StatusXnnErrorBiasBytes }, // BiasOperandIndex
            { Gna2StatusXnnErrorPwlSegments, Gna2StatusXnnErrorOutputBytes }, // PwlOperandIndex
            { Gna2StatusXnnErrorBiasVolume, Gna2StatusXnnErrorBiasBytes }, // WeightScaleFactorOperandIndex
        };
        return errors[operand];
    }

    template<Gna2DeviceGeneration generation, uint32_t operandIndex,
    typename ... T>
    static auto MakeModes(T ... modes)
    {
        UNREFERENCED_PARAMETER(sizeof...(modes));
        return std::pair<Gna2DeviceGeneration, DataModeLimits>{ generation,
            {{ modes... }, GetError<operandIndex>().second } };
    }

    template<Gna2DeviceGeneration generation, uint32_t operandIndex>
        static auto MakeModes(const std::vector<DataMode>& modes)
    {
        return std::pair<Gna2DeviceGeneration, DataModeLimits>{ generation,
        { modes, GetError<operandIndex>().second } };
    }

    static const auto & GetAllSupportedOperationTypes()
    {
        static const auto SUPPORTED_OPERATIONS_TYPES = std::initializer_list<nn_operation>
        {
            INTEL_AFFINE,
            INTEL_AFFINE_DIAGONAL,
            INTEL_AFFINE_MULTIBIAS,
            INTEL_CONVOLUTIONAL,
            INTEL_CONVOLUTIONAL_2D,
            INTEL_CONVOLUTIONAL_1D,
            INTEL_COPY,
            INTEL_DEINTERLEAVE,
            INTEL_GMM,
            INTEL_INTERLEAVE,
            INTEL_RECURRENT,
        };
        return SUPPORTED_OPERATIONS_TYPES;
    }

    template<uint32_t operand, typename ... OperationType>
    static FullCapabilitiesMap MakeFullCaps(OperationType ... operations)
    {
        auto const supportedOperations = std::initializer_list<nn_operation>{ operations... };
        auto const & effective = (sizeof...(OperationType) == 0) ?
            GetAllSupportedOperationTypes() : supportedOperations;
        FullCapabilitiesMap caps;
        for (auto && operation : effective )
        {
            try
            {
                caps.emplace(operation, GetOperands(operation, operand));
            }
            catch (std::out_of_range&)
            {
            }
        }
        return caps;
    }
};

template<uint32_t operandIndex, nn_operation operation>
struct ComponentCaps : protected LayerCapabilities
{
    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        return GetCommonModes(operandIndex, generation);
    }
};

struct LayerCaps : protected LayerCapabilities
{
    template<uint32_t operandIndex, nn_operation operation>
    static const std::shared_ptr<ComponentLimits>& GetComponentLimits(
        Gna2DeviceGeneration generation)
    {
        static const OperationCapabilityMap caps =
        {
            {ComponentCaps<operandIndex, operation>::template Make<Gna2DeviceGeneration0_9>()},
            {ComponentCaps<operandIndex, operation>::template Make<Gna2DeviceGeneration2_0, Gna2DeviceGeneration0_9>()},
            {ComponentCaps<operandIndex, operation>::template Make<Gna2DeviceGeneration3_0>()},
        };
        return caps.at(generation);
    }

    template<uint32_t operandIndex, nn_operation operation>
    static FullCapabilitiesMap::value_type
        MakeAllGensSame()
    {
        return {
            operation,
            {
                Make<Gna2DeviceGeneration0_9, operandIndex, operation>(),
                Make<Gna2DeviceGeneration3_0, operandIndex, operation>(),
            }
        };
    }

    template<Gna2DeviceGeneration generation, uint32_t operandIndex, nn_operation operation>
    static std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation, GetComponentLimits<operandIndex, operation>(generation) };
    }


    template<Gna2DeviceGeneration generation>
    static std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>
    Make(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes)
    {
        return {
            generation,
            std::make_shared<TensorLimits>(order, dimensions, modes)
        };
    }

    template<Gna2DeviceGeneration generation, uint32_t operandIndex, Gna2DeviceGeneration modeGeneration = generation,
    nn_operation operation>
    static std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>
    Make(const OrderLimits order, const ShapeLimits&& dimensions)
    {
        return Make<generation>(order, dimensions, ComponentCaps<operandIndex, operation>::GetModes(modeGeneration));
    }

    template<Gna2DeviceGeneration generation, uint32_t operandIndex, gna_tensor_order order,
    nn_operation operation, typename ... T>
    static std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>
    Make(T ... dimensions)
    {
        return Make<generation, operandIndex, generation, order, operation>(std::forward<T>(dimensions)...);
    }

    template<Gna2DeviceGeneration generation, uint32_t operandIndex,
    Gna2DeviceGeneration modeGeneration, gna_tensor_order order,
    nn_operation operation, typename ... T>
    static auto
    Make(T ... dimensions)
    {
        auto tmpLimits = std::vector<uint32_t>{};
        for (auto && limit : std::vector<StaticCaps>{ { std::forward<T>(static_cast<StaticCaps>(dimensions))... } })
        {
            for (auto && limit1 : limit)
            {
                tmpLimits.push_back(limit1);
            }
        }

        return std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>{
            generation,
            std::make_shared<TensorLimits>(
                order,
                MakeShapeLimits<GetError<operandIndex>().first>(tmpLimits, order),
                ComponentCaps<operandIndex, operation>::GetModes(modeGeneration))
        };
    }

    template<Gna2DeviceGeneration generation, gna_tensor_order order, uint32_t operandIndex>
    static auto MakeCaps(const std::vector<uint32_t>& limits, const std::vector<DataMode>& modes)
    {
        return MakeCaps<generation, order, operandIndex>(limits,
            { modes, GetError<operandIndex>().second });
    }

    template<Gna2DeviceGeneration generation, gna_tensor_order order, uint32_t operandIndex>
    static auto MakeCaps(const std::vector<uint32_t>& limits, const DataModeLimits& modes)
    {
        return Make<generation>(
            { order },
            MakeShapeLimits<GetError<operandIndex>().first>(limits, order),
            modes);
    }
};

template<nn_operation operation>
struct ComponentCaps<InputOperandIndex, operation> : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation, std::make_shared<TensorLimits>(TensorLimits{
               {GNA_TENSOR_HW},
               {{GNA_DIM_H, MakeLimitsMulti<LegacyInputs, InputOperandIndex>()},
               {GNA_DIM_W, MakeLimits<InputGroupMax, InputOperandIndex>()}},
               GetCommonModes(InputOperandIndex, modeGeneration)}) };
    }

    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        return GetCommonModes(InputOperandIndex, generation);
    }
};

template<nn_operation operation>
struct ComponentCaps<OutputOperandIndex, operation> : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation, std::make_shared<TensorLimits>(TensorLimits{
           {GNA_TENSOR_HW},
           {{GNA_DIM_H, MakeLimits<Input, OutputOperandIndex>()},
           {GNA_DIM_W, MakeLimits<InputGroupMax, OutputOperandIndex>()}},
           GetCommonModes(OutputOperandIndex, modeGeneration)}) };
    }

    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        return GetCommonModes(OutputOperandIndex, generation);
    }
};

}
