/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Layer.h"

#include "common.h"

namespace GNA
{
class BaseValidator;

class ConvolutionalLayer2D : public Layer
{
public:
    template<class T>
    ConvolutionalLayer2D(const T& layer, const BaseValidator& validatorIn) :
        Layer(layer, validatorIn, { ConvolutionalTransform2D, ActivationTransform, PoolingTransform2D }, BaseAddress())
    {
        Init();
    }

    virtual ~ConvolutionalLayer2D() = default;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

protected:
    virtual DataConfig GetDataMode() const override;
    void Init();
};

}
