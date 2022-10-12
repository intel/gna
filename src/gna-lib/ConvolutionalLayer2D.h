/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Layer.h"

namespace GNA
{
class BaseValidator;

class ConvolutionalLayer2D : public Layer
{
public:
    ConvolutionalLayer2D(const Gna2Operation& operation, const LayerValidator& validatorIn) :
        Layer(operation, validatorIn, { ConvolutionalTransform2D, ActivationTransform, PoolingTransform2D }, BaseAddress())
    {
        Init();
    }

    virtual ~ConvolutionalLayer2D() = default;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    static std::unique_ptr<const Component> CreateComponentFromParameter(const Shape& shape,
        const LayerValidator& validator, const uint32_t parameterIndex);

protected:
    void Init();
    void Validate3_0ExtraLimits() const;
};

}
