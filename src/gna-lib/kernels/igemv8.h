/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "KernelArguments.h"
#include "KernelMacros.h"

#define AffineKernelImpl1B KERNEL(AffineKernelImpl1B)
#define AffineActiveListKernelImpl1B KERNEL(AffineActiveListKernelImpl1B)
#define AffineMultiBiasKernelImpl1B KERNEL(AffineMultiBiasKernelImpl1B)
#define RecurrentKernelImpl1B KERNEL(RecurrentKernelImpl1B)
#define DiagonalKernelImpl1B KERNEL(DiagonalKernelImpl1B)

#define AffineKernelImpl1B1B KERNEL(AffineKernelImpl1B1B)
#define AffineActiveListKernelImpl1B1B KERNEL(AffineActiveListKernelImpl1B1B)
#define AffineMultiBiasKernelImpl1B1B KERNEL(AffineMultiBiasKernelImpl1B1B)
#define RecurrentKernelImpl1B1B KERNEL(RecurrentKernelImpl1B1B)
#define DiagonalKernelImpl1B1B KERNEL(DiagonalKernelImpl1B1B)
#define TransposeKernelImpl1B KERNEL(TransposeKernelImpl1B)

#define AffineKernelImpl1B2B KERNEL(AffineKernelImpl1B2B)
#define AffineActiveListKernelImpl1B2B KERNEL(AffineActiveListKernelImpl1B2B)
#define AffineMultiBiasKernelImpl1B2B KERNEL(AffineMultiBiasKernelImpl1B2B)
#define RecurrentKernelImpl1B2B KERNEL(RecurrentKernelImpl1B2B)
#define DiagonalKernelImpl1B2B KERNEL(DiagonalKernelImpl1B2B)
#define TransposeKernelImpl2B KERNEL(TransposeKernelImpl2B)

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
void AffineKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// uses active outputs list
void AffineActiveListKernelImpl1B(
    ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// handles multi bias
void AffineMultiBiasKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config);

// Calculates recurrent transform on flat input vectors
// (input vectors in N rows, vector elements in K columns)
void RecurrentKernelImpl1B(ExecutionKernelConfig<RecurrentConfig> const * const config);

void DiagonalKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config);

void TransposeKernelImpl2B(TransposeConfig const * const transposeConfig);

#if OPT_LEVEL < 2
void TransposeKernelImpl1B(TransposeConfig const * const transposeConfig);
void AffineKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineActiveListKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void AffineActiveListKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void AffineMultiBiasKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineMultiBiasKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config);
void RecurrentKernelImpl1B1B(ExecutionKernelConfig<RecurrentConfig> const * const config);
void RecurrentKernelImpl1B2B(ExecutionKernelConfig<RecurrentConfig> const * const config);
void DiagonalKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void DiagonalKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config);
#endif

#if OPT_LEVEL == 2 || OPT_LEVEL == 3 || OPT_LEVEL == 6 || OPT_LEVEL == 7
void TransposeKernelImpl1B(TransposeConfig const * const transposeConfig);
#endif

#if OPT_LEVEL == 3 || OPT_LEVEL == 7
void AffineKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineMultiBiasKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineActiveListKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void RecurrentKernelImpl1B1B(ExecutionKernelConfig<RecurrentConfig> const * const config);
#endif
