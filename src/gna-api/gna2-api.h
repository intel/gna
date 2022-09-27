/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

/**************************************************************************//**
 @file gna2-api.h
 @brief Gaussian and Neural Accelerator (GNA) 3.0 API.
 @nosubgrouping

 ******************************************************************************

 @mainpage GNA 3.0 Introduction

    GNA-3.0 introduces acceleration for both Gaussian-Mixture-Model (GMM)
    and Neural-Networks (xNN) groups of algorithms used by different speech
    recognition operations as well as sensing. The GNA supports both GMM
    and xNN operations. GNA can be activated to perform a sequence of basic
    operations which can be any of the GMM or xNN operations and/or additional
    helper functions. These operations are organized in layers which define
    the operation and its properties.

    The GNA-3.0 IP module scalable and configurable, providing and option
    to tuned GNA HW for various algorithms and use-cases. GNA can be tuned
    to optimize Large-Vocabulary Speech-Recognition algorithms which require
    relative large compute power, or be tuned for low-power always-on sensing
    algorithms. GNA-3.0 extends its support for use-cases beyond speech
    such as low-power always-on sensing, therefore it is not limited to these,
    and may be used by other algorithms.

 ******************************************************************************

 @addtogroup GNA2_API Gaussian and Neural Accelerator (GNA) 3.0 API.

 Gaussian mixture models and Neural network Accelerator.

 @note
 API functions are assumed NOT thread-safe until stated otherwise.

 @{
 *****************************************************************************/

#ifndef __GNA2_API_H
#define __GNA2_API_H

#include "gna2-common-api.h"
#include "gna2-device-api.h"
#include "gna2-inference-api.h"
#include "gna2-instrumentation-api.h"
#include "gna2-memory-api.h"
#include "gna2-model-api.h"

#include <cstdint>

#endif // __GNA2_API_H

/**
 @}
 */
