/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{

enum SubModelType
{
    Software,
    Hardware,
    GMMHardware
};

class SubModel
{
public:

    SubModel(SubModelType type, uint32_t layerIndex);
    SubModel(SubModel&& rhs) = default;
    SubModel(const SubModel &) = delete;
    SubModel& operator=(const SubModel&) = delete;

    uint32_t GetLayerCount() const;
    void AddLayer();

    bool Contains(uint32_t layerIndex) const
    {
        return layerIndex >= LayerIndex && layerIndex < LayerIndex + GetLayerCount();
    }

    static bool IsSoftwareLayer(uint32_t layerIndex,
        const std::vector<std::unique_ptr<SubModel>>& subModels);

    const SubModelType Type;
    const uint32_t LayerIndex;

private:
    uint32_t layerCount;
};

}
