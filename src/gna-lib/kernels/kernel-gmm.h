/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "KernelMacros.h"

#include <cstdint>

#define gmm_maxmix_8u8u_32u     KERNEL(gmm_maxmix_8u8u_32u)
#define gmm_maxmix_8u16u_32u    KERNEL(gmm_maxmix_8u16u_32u)
#if OPT_LEVEL > 1
#define gmm_maxmix_8u8u_32u_g1  KERNEL(gmm_maxmix_8u8u_32u_g1)
#define gmm_maxmix_8u8u_32u_g2  KERNEL(gmm_maxmix_8u8u_32u_g2)
#define gmm_maxmix_8u8u_32u_g3  KERNEL(gmm_maxmix_8u8u_32u_g3)
#define gmm_maxmix_8u8u_32u_g4  KERNEL(gmm_maxmix_8u8u_32u_g4)
#define gmm_maxmix_8u8u_32u_g5  KERNEL(gmm_maxmix_8u8u_32u_g5)
#define gmm_maxmix_8u8u_32u_g6  KERNEL(gmm_maxmix_8u8u_32u_g6)
#define gmm_maxmix_8u8u_32u_g7  KERNEL(gmm_maxmix_8u8u_32u_g7)
#define gmm_maxmix_8u8u_32u_g8  KERNEL(gmm_maxmix_8u8u_32u_g8)
#endif

struct GmmConfig;


void gmm_maxmix_8u8u_32u(GmmConfig const * const config);

void gmm_maxmix_8u16u_32u(GmmConfig const * const config);

#if OPT_LEVEL > 1
void gmm_maxmix_8u8u_32u_g1(GmmConfig const * const config);

void gmm_maxmix_8u8u_32u_g2(GmmConfig const * const config);

void gmm_maxmix_8u8u_32u_g3(GmmConfig const * const config);

void gmm_maxmix_8u8u_32u_g4(GmmConfig const * const config);

void gmm_maxmix_8u8u_32u_g5(GmmConfig const * const config);

void gmm_maxmix_8u8u_32u_g6(GmmConfig const * const config);

void gmm_maxmix_8u8u_32u_g7(GmmConfig const * const config);

void gmm_maxmix_8u8u_32u_g8(GmmConfig const * const config);
#endif
