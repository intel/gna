/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "LayerCapabilities.h"

namespace GNA
{

struct GmmLayerCapabilities : LayerCapabilities
{
    static const FullCapabilitiesMap& GetOperands(uint32_t operandIndex);
};

/** Maximum number of mixture components per GMM State */
constexpr uint32_t GMM_MIXTURE_COMP_COUNT_MAX = 4096;

/** Maximum number of GMM states, active list elements and  */
constexpr uint32_t GMM_STATES_COUNT_MAX = 262144;

/** Size of memory alignment for mean, variance vectors and Gaussian Constants */
constexpr uint32_t GMM_MEM_ALIGNMENT = 8;

/** Mean vector width in bytes */
constexpr uint32_t GMM_MEAN_VALUE_SIZE = 1;

/** Minimum variance vector width in bytes */
constexpr uint32_t GMM_COVARIANCE_SIZE_MIN = 1;

/** Maximum variance vector width in bytes */
constexpr uint32_t GMM_COVARIANCE_SIZE_MAX = 2;

/** Score width in bytes */
constexpr uint32_t GMM_SCORE_SIZE = 4;

/** Minimum length of a vector */
constexpr uint32_t GMM_FV_ELEMENT_COUNT_MIN = 24;

/** The allowed alignment of vector lengths */
constexpr uint32_t GMM_FV_ELEMENT_COUNT_MULTIPLE_OF = 8;

/** Feature vector width in bytes */
constexpr uint32_t GMM_FV_ELEMENT_SIZE = 1;

/** Maximum length of a vector */
constexpr uint32_t GMM_FV_ELEMENT_COUNT_MAX = 96;

/** Gaussian Constants width in bytes */
constexpr uint32_t GMM_CONSTANTS_SIZE = 4;

}
