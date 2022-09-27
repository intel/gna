/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Tensor.h"

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;

struct LayerOutput : public Tensor
{
    LayerOutput(const Gna2Operation &operation, const LayerValidator& validatorIn);
    virtual ~LayerOutput() = default;

    const Tensor ScratchPad;
    const uint32_t Grouping;
    const uint32_t ElementCount;

    static void * getScratchpadForOperation(const nn_operation &operation);

protected:
    static const FullCapabilitiesMap capabilities;

    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(
        const Gna2Operation& operation, const LayerValidator& validatorIn) const override;

private:
    static ApiShape GetShape(const Gna2Operation & operation);
};

}
