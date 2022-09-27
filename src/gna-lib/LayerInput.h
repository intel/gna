/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Shape.h"
#include "Tensor.h"

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;

struct LayerInput : public Tensor
{
    LayerInput(const Gna2Operation &operation, const LayerValidator& validatorIn);
    virtual ~LayerInput() = default;

    static bool IsInputInterleave(const Gna2Tensor &apiTensor,
        const BaseValidator& validatorIn);

    const uint32_t Grouping;
    const uint32_t ElementCount;

protected:
    static const FullCapabilitiesMap capabilities;
    static ApiShape GetShape(const Gna2Operation & operation);

    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(
        const Gna2Operation& operation, const LayerValidator& validatorIn) const override;
};

}
