/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#if defined(__GNUC__)
#include <cpuid.h>
static inline unsigned long long _xgetbv(unsigned int ctr)
{
    int a, d;
    __asm("xgetbv" : "=a"(a),"=d"(d) : "c"(ctr) : );
    return static_cast<unsigned long long>(a) | ((static_cast<unsigned long long>(d)) << 32);
}
#define cpuid(info, level) __cpuid_count(level, 0, info[0], info[1], info[2], info[3])
#else
#if defined(__INTEL_COMPILER)
#include <immintrin.h>
#else // __INTEL_COMPILER
#include <intrin.h>
#endif
#define cpuid(info, level) __cpuidex((int*)(info), level, 0)
#endif // __GNUC__

#include "gmm.h"

#include "AccelerationDetector.h"
#include "Logger.h"

#include "gna2-inference-impl.h"

#include <map>
#include <memory>

using namespace GNA;

/**
 * Masks for CPU extensions detection
 */
#define SSE4_MASK 0x00180000  // mask for SSE4_1, SSE4_2 feature flags, 19,20 bits
#define AVX1_MASK 0x18000000  // mask for OSXSAVE and AVX feature flags, 27,28 bits
#define AVX2_MASK 0x00000020  // mask for AVX2 feature flag, 5 bit
#define XYMM_MASK 0x00000006  // mask for OS enabled XMM+YMM state support flag, 1,2 bits
#define AVX2_MASK_ASM 000000020H  // mask for AVX2 feature flag, 5 bit

/**
 * If _XCR_XFEATURE_ENABLED_MASK is not defined set it to 0
 * intrin.h header file containing this flag is MS-specific
 * and not available on linux platforms
 */
#ifndef _XCR_XFEATURE_ENABLED_MASK
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

static const AccelerationMode GNA_GEN_SAT {Gna2AccelerationModeGeneric, true };
static const AccelerationMode GNA_GEN_FAST {Gna2AccelerationModeGeneric, false };
static const AccelerationMode GNA_SSE4_2_SAT {Gna2AccelerationModeSse4x2, true };
static const AccelerationMode GNA_SSE4_2_FAST {Gna2AccelerationModeSse4x2, false };
static const AccelerationMode GNA_AVX1_SAT {Gna2AccelerationModeAvx1, true };
static const AccelerationMode GNA_AVX1_FAST {Gna2AccelerationModeAvx1, false };
static const AccelerationMode GNA_AVX2_SAT {Gna2AccelerationModeAvx2, true };
static const AccelerationMode GNA_AVX2_FAST{ Gna2AccelerationModeAvx2, false };

static const AccelerationMode GNA_SW_SAT { Gna2AccelerationModeSoftware,true };
static const AccelerationMode GNA_SW_FAST { Gna2AccelerationModeSoftware,false };
static const AccelerationMode GNA_AUTO_FAST { Gna2AccelerationModeAuto,false };
static const AccelerationMode GNA_AUTO_SAT { Gna2AccelerationModeAuto,true };

std::map<kernel_op, std::map<KernelMode, KernelMap<VoidKernel>>>
AccelerationDetector::Kernels = {
    { KERNEL_AFFINE,
    {
        {
            {GNA_INT16, GNA_INT8, GNA_BIAS_MODE_RICH_FORMAT },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B2Bfull) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B2Bfull) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.affineSingle1Bfull) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.affineSingle1Bfull) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.affineSingle1Bfull) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.affineSingle1Bfull) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.affineSingle1Bfull) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.affineSingle1Bfull) } }

            }
        },
        {
            { GNA_INT16, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B2Bfull) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B2Bfull) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.affineSingle2Bfull) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.affineSingle2Bfull) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.affineSingle2Bfull) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.affineSingle2Bfull) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.affineSingle2Bfull) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.affineSingle2Bfull) } }
            }
        },
        {
            { GNA_INT8, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B1Bfull) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B1Bfull) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B1Bfull) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B1Bfull) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B1Bfull) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B1Bfull) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B1Bfull) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B1Bfull) } }

            }
        },
        {
            { GNA_INT8, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B1Bfull) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B1Bfull) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B1Bfull) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B1Bfull) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B1Bfull) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B1Bfull) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B1Bfull) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B1Bfull) } }
            }
        }
    }
    },
    { KERNEL_AFFINE_AL,
    {
        {
            { GNA_INT16, GNA_INT8, GNA_BIAS_MODE_RICH_FORMAT },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B2Bal) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B2Bal) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.affineSingle1Bal) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.affineSingle1Bal) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.affineSingle1Bal) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.affineSingle1Bal) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.affineSingle1Bal) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.affineSingle1Bal) } }
            }
        },
        {
            { GNA_INT16, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B2Bal) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B2Bal) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.affineSingle2Bal) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.affineSingle2Bal) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.affineSingle2Bal) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.affineSingle2Bal) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.affineSingle2Bal) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.affineSingle2Bal) } }
            }
        },
        {
            { GNA_INT8, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B1Bal) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B1Bal) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B1Bal) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B1Bal) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B1Bal) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B1Bal) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle1B1Bal) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle1B1Bal) } }
            }
        },
        {
            { GNA_INT8, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B1Bal) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B1Bal) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B1Bal) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B1Bal) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B1Bal) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B1Bal) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineSingle2B1Bal) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineSingle2B1Bal) } }
            }
        }
    }
    },
    { KERNEL_AFFINE_MULTIBIAS,
    {
        {
            { GNA_INT16, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti1B2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti1B2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.affineMulti1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.affineMulti1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.affineMulti1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.affineMulti1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.affineMulti1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.affineMulti1B) } }
            }
        },
        {
            { GNA_INT16, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti2B2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti2B2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.affineMulti2B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.affineMulti2B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.affineMulti2B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.affineMulti2B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.affineMulti2B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.affineMulti2B) } }
            }
        },
        {
            { GNA_INT8, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti1B1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti1B1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti1B1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti1B1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti1B1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti1B1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti1B1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti1B1B) } }
            }
        },
        {
            { GNA_INT8, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti2B1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti2B1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti2B1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti2B1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti2B1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti2B1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.affineMulti2B1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.affineMulti2B1B) } }
            }
        }
    }
    },
    { KERNEL_RECURRENT,
    {
        {
            { GNA_INT16, GNA_INT8, GNA_BIAS_MODE_RICH_FORMAT },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent1B2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent1B2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.recurrent1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.recurrent1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent1B2B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent1B2B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent1B2B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent1B2B) } }
            }
        },
        {
            { GNA_INT16, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent2B2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent2B2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.recurrent2B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.recurrent2B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent2B2B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent2B2B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent2B2B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent2B2B) } }
            }
        },
        {
            { GNA_INT8, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent1B1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent1B1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent1B1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent1B1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent1B1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent1B1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent1B1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent1B1B) } }
            }
        },
        {
            { GNA_INT8, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent2B1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent2B1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent2B1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent2B1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent2B1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent2B1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.recurrent2B1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.recurrent2B1B) } }
            }
        },
    }
    },
    { KERNEL_AFFINE_DIAGONAL,
    {
        {
            { GNA_INT16, GNA_INT8, GNA_BIAS_MODE_RICH_FORMAT },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal1B2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal1B2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.diagonal1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.diagonal1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.diagonal1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.diagonal1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.diagonal1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.diagonal1B) } }
            }
        },
        {
            { GNA_INT16, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal2B2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal2B2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.diagonal2B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.diagonal2B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.diagonal2B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.diagonal2B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.diagonal2B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.diagonal2B) } }
            }
        },
        {
            { GNA_INT8, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal1B1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal1B1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal1B1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal1B1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal1B1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal1B1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal1B1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal1B1B) } }
            }
        },
        {
            { GNA_INT8, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal2B1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal2B1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal2B1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal2B1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal2B1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal2B1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.diagonal2B1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.diagonal2B1B) } }
            }
        },
    }
    },
    { KERNEL_TRANSPOSE,
    {
        {
            { GNA_INT8},
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.transpose1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.transpose1B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.transpose1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.transpose1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.transpose1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.transpose1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.transpose1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.transpose1B) } }
            }
        },
        {
            { GNA_INT16},
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.transpose2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.transpose2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.transpose) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.transpose) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.transpose2B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.transpose2B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.transpose2B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.transpose2B) } }
            }
        }
    }
    },
    { KERNEL_COPY,
    {
        {
            { GNA_INT8 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.copy1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.copy1B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.copy1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.copy1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.copy1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.copy1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.copy1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.copy1B) } }
            }
        },
        {
            { GNA_INT16 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.copy2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.copy2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.copy) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.copy) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.copy) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.copy) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.copy) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.copy) } }
            }
        }
    }
    },
    { KERNEL_CONVOLUTIONAL,
    {
        {
            { GNA_INT16 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.convolution) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.convolution) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.convolution) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.convolution) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.convolution) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.convolution) } }
            }
        },
        {
            { GNA_INT8 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution1B) } }
            }
        }
    }
    },
    { KERNEL_CONVOLUTIONAL_2D,
    {
        {
            { GNA_INT16, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D1B2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D1B2B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D1B2B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D1B2B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D1B2B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D1B2B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D1B2B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D1B2B) } }
            }
        },
        {
            { GNA_INT16, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D2B2B) } },
               { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D2B2B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D2B2B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D2B2B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D2B2B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D2B2B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D2B2B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D2B2B) } }
            }
        },
        {
            { GNA_INT8, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D1B1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D1B1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D1B1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D1B1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D1B1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D1B1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D1B1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D1B1B) } }
            }
        },
        {
            { GNA_INT8, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D2B1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D2B1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D2B1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D2B1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D2B1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D2B1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolution2D2B1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolution2D2B1B) } }
            }
        }
    }
    },
    { KERNEL_POOLING,
    {
        {
            { GNA_INT16, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2B) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.convolutionPooling) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.convolutionPooling) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.convolutionPooling) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.convolutionPooling) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.convolutionPooling) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.convolutionPooling) } }
            }
        },
        {
            { GNA_INT8, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling1B) } }
            }
        }
    }
    },
    { KERNEL_POOLING_2D,
    {
        {
            { GNA_INT8 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D1B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D1B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D1B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D1B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D1B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D1B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D1B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D1B) } }
            }
        },
        {
            { GNA_INT16 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D2B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D2B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D2B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D2B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D2B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D2B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D2B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D2B) } }
            }
        },
        {
            { GNA_INT32 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D4B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D4B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D4B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D4B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D4B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D4B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D4B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D4B) } }
            }
        },
         {
            { GNA_DATA_ACTIVATION_DISABLED },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D4B) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D4B) } },

                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D4B) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D4B) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D4B) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D4B) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.convolutionPooling2D4B) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.convolutionPooling2D4B) } }
            }
        },
    }
    },
    { KERNEL_PWL,
    {
        {
            { GNA_INT16 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_generic_sat.pwl) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_generic.pwl) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4_sat.pwl) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_sse4.pwl) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1_sat.pwl) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx1.pwl) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2_sat.pwl) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(xnnKernel_avx2.pwl) } }
            }
        }
    }
    },
    { KERNEL_GMM,
    {
        {
            { GNA_UINT8, GNA_UINT8, GNA_UINT32 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_generic.gmmMaxMix8) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_generic.gmmMaxMix8) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_sse4.gmmMaxMix8) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_sse4.gmmMaxMix8) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_avx1.gmmMaxMix8) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_avx1.gmmMaxMix8) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_avx2.gmmMaxMix8) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_avx2.gmmMaxMix8) } }
            }
        },
        {
            { GNA_UINT8, GNA_UINT16, GNA_UINT32 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_generic.gmmMaxMix16) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_generic.gmmMaxMix16) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_sse4.gmmMaxMix16) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_sse4.gmmMaxMix16) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_avx1.gmmMaxMix16) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_avx1.gmmMaxMix16) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_avx2.gmmMaxMix16) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_avx2.gmmMaxMix16) } }
            }
        }
    }
    },
    { KERNEL_GMM_AL,
    {
        {
            { GNA_UINT8, GNA_UINT8, GNA_UINT32 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_generic.gmmMaxMix8ActiveList) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_generic.gmmMaxMix8ActiveList) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_sse4.gmmMaxMix8ActiveList) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_sse4.gmmMaxMix8ActiveList) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_avx1.gmmMaxMix8ActiveList) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_avx1.gmmMaxMix8ActiveList) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_avx2.gmmMaxMix8ActiveList) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_avx2.gmmMaxMix8ActiveList) } }
            }
        },
        {
            { GNA_UINT8, GNA_UINT16, GNA_UINT32 },
            {
                { { GNA_GEN_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_generic.gmmMaxMix16ActiveList) } },
                { { GNA_GEN_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_generic.gmmMaxMix16ActiveList) } },
                { { GNA_SSE4_2_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_sse4.gmmMaxMix16ActiveList) } },
                { { GNA_SSE4_2_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_sse4.gmmMaxMix16ActiveList) } },

                { { GNA_AVX1_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_avx1.gmmMaxMix16ActiveList) } },
                { { GNA_AVX1_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_avx1.gmmMaxMix16ActiveList) } },
                { { GNA_AVX2_SAT },{ reinterpret_cast<VoidKernel>(gmmKernel_avx2.gmmMaxMix16ActiveList) } },
                { { GNA_AVX2_FAST },{ reinterpret_cast<VoidKernel>(gmmKernel_avx2.gmmMaxMix16ActiveList) } }
            }
        }
    }
    }
};

AccelerationDetector::AccelerationDetector()
{
    DetectSoftwareAccelerationModes();
}

void AccelerationDetector::DetectSoftwareAccelerationModes()
{
    supportedCpuAccelerations = { Gna2AccelerationModeGeneric };
    // generic, fastest software and auto always supported
    accelerationModes[GNA_GEN_SAT] = true;
    accelerationModes[GNA_GEN_FAST] = true;
    accelerationModes[GNA_SW_SAT] = true;
    accelerationModes[GNA_SW_FAST] = true;
    accelerationModes[GNA_AUTO_SAT] = true;
    accelerationModes[GNA_AUTO_FAST] = true;

    unsigned int cpuId[4];           // cpu id string
    unsigned long long xcrFeature = 0;

    cpuid(cpuId, 0);
    auto largestFunctionId = cpuId[0];

    // get CPU IDs
    cpuid(cpuId, 1);

    // detect SSE4
    // check both SSE4_1, SSE4_2 feature flags (bits 19,20)
    if ((cpuId[2] & SSE4_MASK) == SSE4_MASK)
    {
        accelerationModes[GNA_SSE4_2_FAST] = true;
        accelerationModes[GNA_SSE4_2_SAT] = true;
        supportedCpuAccelerations.push_back(Gna2AccelerationModeSse4x2);
    }

    if ((cpuId[2] & AVX1_MASK) == AVX1_MASK)
    {
        // check OS has enabled both XMM and YMM state support
        xcrFeature = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        xcrFeature = xcrFeature & XYMM_MASK;
        if (XYMM_MASK == xcrFeature)
        {
            accelerationModes[GNA_AVX1_FAST] = true;
            accelerationModes[GNA_AVX1_SAT] = true;
            supportedCpuAccelerations.push_back(Gna2AccelerationModeAvx1);
        }

        // check AVX2 flag
        if (largestFunctionId >= 7)
        {
            cpuid(cpuId, 7);
            if ((cpuId[1] & AVX2_MASK) == AVX2_MASK)
            {
                accelerationModes[GNA_AVX2_FAST] = true;
                accelerationModes[GNA_AVX2_SAT] = true;
                supportedCpuAccelerations.push_back(Gna2AccelerationModeAvx2);
            }
        }
    }
}

void AccelerationDetector::PrintAllAccelerationModes() const
{
    for (const auto& modeState : accelerationModes)
    {
        const auto name = modeState.first.GetName();
        Log->Message("%s\t%s\n", name, modeState.second ? "Yes" : "No");
    }
}

const std::vector<Gna2AccelerationMode>& AccelerationDetector::GetSupportedCpuAccelerations() const
{
    return supportedCpuAccelerations;
}
