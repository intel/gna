/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <cstdint>

namespace GNA
{

enum SubmodelType
{
    Software,
    Hardware,
    GMMHardware
};

class SubModel
{
public:

    SubModel(SubmodelType type, uint32_t layerIndex);
    SubModel(SubModel&& rhs) = default;
    SubModel(const SubModel &) = delete;
    SubModel& operator=(const SubModel&) = delete;

    uint32_t GetLayerCount() const;
    void AddLayer();

    const SubmodelType Type;
    const uint32_t LayerIndex;

private:
    uint32_t layerCount;
};

}
