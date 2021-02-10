/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "KernelMacros.h"
#include "KernelArguments.h"
#include "ConvolutionKernelArguments.h"
#include "PoolingKernelArguments.h"

#include <limits>

#define ConvolutionKernelImpl KERNEL(ConvolutionKernelImpl)
#define ConvolutionPoolingKernelImpl KERNEL(ConvolutionPoolingKernelImpl)
#define ConvolutionKernelImpl1B KERNEL(ConvolutionKernelImpl1B)
#define ConvolutionPoolingKernelImpl1B KERNEL(ConvolutionPoolingKernelImpl1B)
#define ConvolutionKernelImpl2B KERNEL(ConvolutionKernelImpl2B)
#define ConvolutionPoolingKernelImpl2B KERNEL(ConvolutionPoolingKernelImpl2B)
#define Pooling2DKernelImpl2B KERNEL(Pooling2DKernelImpl2B)
#define MaxPartialPoolingFunction KERNEL(MaxPartialPoolingFunction)
#define SumPartialPoolingFunction KERNEL(SumPartialPoolingFunction)

#define Convolution2DKernelImpl1B1B KERNEL(Convolution2DKernelImpl1B1B)
#define Convolution2DKernelImpl1B2B KERNEL(Convolution2DKernelImpl1B2B)
#define Convolution2DKernelImpl2B1B KERNEL(Convolution2DKernelImpl2B1B)
#define Convolution2DKernelImpl2B2B KERNEL(Convolution2DKernelImpl2B2B)

#define Pooling2DKernelImpl1B KERNEL(Pooling2DKernelImpl1B)
#define Pooling2DKernelImpl2B KERNEL(Pooling2DKernelImpl2B)
#define Pooling2DKernelImpl4B KERNEL(Pooling2DKernelImpl4B)

using GNA::PwlCached;

#ifdef __cplusplus
extern "C" {
#endif

    void ConvolutionKernelImpl(ConvolutionConfig const * const filterConfig);

    void ConvolutionPoolingKernelImpl(ConvolutionConfig const * const filterConfig,
        PoolingConfig const * const poolConfig, PwlCached const * const pwl);

#if OPT_LEVEL < 2
    void ConvolutionKernelImpl1B(ConvolutionConfig const * const filterConfig);
    void ConvolutionKernelImpl2B(ConvolutionConfig const * const filterConfig);
    void ConvolutionPoolingKernelImpl1B(ConvolutionConfig const * const filterConfig,
        PoolingConfig const * const poolConfig, PwlCached const * const pwl);
    void ConvolutionPoolingKernelImpl2B(ConvolutionConfig const * const filterConfig,
        PoolingConfig const * const poolConfig, PwlCached const * const pwl);
    void Convolution2DKernelImpl1B1B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config);
    void Convolution2DKernelImpl1B2B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config);
    void Convolution2DKernelImpl2B1B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config);
    void Convolution2DKernelImpl2B2B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config);
    void Pooling2DKernelImpl1B(ExecutionKernelConfig<PoolingConfig2D> const * const config);
    void Pooling2DKernelImpl2B(ExecutionKernelConfig<PoolingConfig2D> const * const config);
    void Pooling2DKernelImpl4B(ExecutionKernelConfig<PoolingConfig2D> const * const config);
#endif
/* Calculates MaxPartialPoolingFunction
* @PS   number of pool size
* @PNE  number of pool entries
* @PSI  number of pool start index
* @P    pointer to pool array
* @V    pointer to value
*/
void MaxPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t* P, int64_t* V);


/* Calculates SumPartialPoolingFunction
* @PS   number of pool size
* @PNE  number of pool entries
* @PSI  number of pool start index
* @P    pointer to pool array
* @V    pointer to value
*/
void SumPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t* P, int64_t* V);

#ifdef __cplusplus
}
#endif

template<class T = int32_t>
__forceinline void gna_saturate_cast(int64_t & val, uint32_t & saturationCount)
{
    if (val > (std::numeric_limits<T>::max)())
    {
        saturationCount++;
        val = (std::numeric_limits<T>::max)();
    }
    else if (val < (std::numeric_limits<T>::min)())
    {
        saturationCount++;
        val = (std::numeric_limits<T>::min)();
    }
}
