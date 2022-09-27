/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "KernelArguments.h"
#include "KernelMacros.h"

#define AffineKernelImpl2B KERNEL(AffineKernelImpl2B)
#define AffineActiveListKernelImpl2B KERNEL(AffineActiveListKernelImpl2B)
#define AffineMultiBiasKernelImpl2B KERNEL(AffineMultiBiasKernelImpl2B)
#define RecurrentKernelImpl2B KERNEL(RecurrentKernelImpl2B)
#define DiagonalKernelImpl2B KERNEL(DiagonalKernelImpl2B)

#define AffineActiveListKernelImpl2B1B KERNEL(AffineActiveListKernelImpl2B1B)
#define RecurrentKernelImpl2B1B KERNEL(RecurrentKernelImpl2B1B)
#define DiagonalKernelImpl2B1B KERNEL(DiagonalKernelImpl2B1B)
#define AffineKernelImpl2B1B KERNEL(AffineKernelImpl2B1B)
#define AffineMultiBiasKernelImpl2B1B KERNEL(AffineMultiBiasKernelImpl2B1B)
#define TransposeKernelImpl1B KERNEL(TransposeKernelImpl1B)

#define AffineActiveListKernelImpl2B2B KERNEL(AffineActiveListKernelImpl2B2B)
#define RecurrentKernelImpl2B2B KERNEL(RecurrentKernelImpl2B2B)
#define DiagonalKernelImpl2B2B KERNEL(DiagonalKernelImpl2B2B)
#define AffineKernelImpl2B2B KERNEL(AffineKernelImpl2B2B)
#define AffineMultiBiasKernelImpl2B2B KERNEL(AffineMultiBiasKernelImpl2B2B)
#define TransposeKernelImpl2B KERNEL(TransposeKernelImpl2B)

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
void AffineKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// uses active outputs list
void AffineActiveListKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// handles multi bias
void AffineMultiBiasKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config);

// Calculates recurrent transform on flat input vectors
// (input vectors in N rows, vector elements in K columns)
void RecurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config);

void DiagonalKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config);

void TransposeKernelImpl2B(TransposeConfig const * const transposeConfig);

#if OPT_LEVEL < 2
void TransposeKernelImpl1B(TransposeConfig const * const transposeConfig);
void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineActiveListKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void AffineActiveListKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void AffineMultiBiasKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineMultiBiasKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config);
void RecurrentKernelImpl2B1B(ExecutionKernelConfig<RecurrentConfig> const * const config);
void RecurrentKernelImpl2B2B(ExecutionKernelConfig<RecurrentConfig> const * const config);
void DiagonalKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void DiagonalKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config);
#endif

#if OPT_LEVEL == 3 || OPT_LEVEL == 7
void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineMultiBiasKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineActiveListKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void RecurrentKernelImpl2B1B(ExecutionKernelConfig<RecurrentConfig> const * const config);
#endif
