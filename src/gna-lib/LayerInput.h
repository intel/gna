/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Shape.h"
#include "Tensor.h"

#include "common.h"
#include "gna-api-types-xnn.h"

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;

struct LayerInput : public Tensor
{
    LayerInput(const nn_layer &layer, const LayerValidator& validatorIn);
    LayerInput(const Gna2Operation &operation, const LayerValidator& validatorIn);
    virtual ~LayerInput() = default;

    static bool IsInputInterleave(const Gna2Tensor &apiTensor,
                       const BaseValidator& validatorIn);

    const uint32_t Grouping;
    const uint32_t ElementCount;

protected:
    static const FullCapabilitiesMap capabilities;

    static Shape GetDimensions(const nn_layer& layer, gna_tensor_order order);

    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(
        const Gna2Operation& operation, const LayerValidator& validatorIn) const override;
    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(const nn_layer& layer) const override;
};

}
