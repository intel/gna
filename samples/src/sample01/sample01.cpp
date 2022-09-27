/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#define __STDC_WANT_LIB_EXT1__ 1

#include "gna2-api.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#include <mm_malloc.h>
#endif

#ifndef __STDC_LIB_EXT1__
#define memcpy_s(_Destination, _DestinationSize, _Source, _SourceSize) memcpy(_Destination, _Source, _SourceSize)
#endif

void HandleGnaStatus(Gna2Status status, const char* where, const char* statusFrom)
{
    if (status != Gna2StatusSuccess)
    {
        auto const size = Gna2StatusGetMaxMessageLength();
        auto msg = std::unique_ptr<char[]>(new char[size]());
        Gna2StatusGetMessage(status, msg.get(), size);

        std::string s = "In: ";
        s += where;
        s += ": FAILURE in ";
        s += statusFrom;
        s += ": Status message: ";
        s += msg.get();
        printf("%s\n", s.c_str());
        exit(static_cast<int32_t>(status));
    }
}

void print_outputs(
    int32_t* outputs,
    uint32_t nRows,
    uint32_t nColumns
)
{
    printf("\nOutputs:\n");
    for (uint32_t i = 0; i < nRows; ++i)
    {
        for (uint32_t j = 0; j < nColumns; ++j)
        {
            printf("%d\t", outputs[i * nColumns + j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

void* customAlloc(uint32_t dumpedModelSize)
{
    if (0 == dumpedModelSize)
    {
        printf("customAlloc has invalid dump model size: %" PRIu32 "\n", dumpedModelSize);
        exit(-Gna2StatusMemorySizeInvalid);
    }
    return _mm_malloc(dumpedModelSize, 4096);
}

void CleanupOrError(Gna2Status status, unsigned deviceIndex, void* memory, uint32_t modelId, uint32_t requestConfigId,
                    const char* where, const char* statusFrom)
{
    if (!Gna2StatusIsSuccessful(status))
    {
        if (static_cast<uint32_t>(GNA2_DISABLED) != requestConfigId)
        {
            Gna2RequestConfigRelease(modelId);
        }
        if (static_cast<uint32_t>(GNA2_DISABLED) != modelId)
        {
            Gna2ModelRelease(modelId);
        }
        if (nullptr != memory)
        {
            Gna2MemoryFree(memory);
        }
        if (static_cast<uint32_t>(GNA2_DISABLED) != modelId)
        {
            Gna2DeviceClose(deviceIndex);
        }
        HandleGnaStatus(status, where, statusFrom);
    }
}

int main(int argc, char* argv[])
try
{
    constexpr uint32_t W = 16;
    constexpr uint32_t H = 8;
    constexpr uint32_t B = 4;
    int16_t weights[H * W] = {
        // sample weight matrix (8 rows, 16 cols)
        -6, -2, -1, -1, -2, 9, 6, 5, 2, 4, -1, 5, -2, -4, 0, 9,
        // in case of affine layer this is the left operand of matrix mul
        -8, 8, -4, 6, 5, 3, -7, -9, 7, 0, -4, -1, 1, 7, 6, -6, // in this sample the numbers are random and meaningless
        2, -8, 6, 5, -1, -2, 7, 5, -1, 4, 8, 7, -9, -1, 7, 1,
        0, -2, 1, 0, 6, -6, 7, 4, -6, 0, 3, -2, 1, 8, -6, -2,
        -6, -3, 4, -2, -8, -6, 6, 5, 6, -9, -5, -2, -5, -8, -6, -2,
        -7, 0, 6, -3, -1, -6, 4, 1, -4, -5, -3, 7, 9, -9, 9, 9,
        0, -2, 6, -3, 5, -2, -1, -3, -5, 7, 6, 6, -8, 0, -4, 9,
        2, 7, -8, -7, 8, -6, -6, 1, 7, -4, -4, 9, -6, -6, 5, -7
    };

    int16_t inputs[W * B] = {
        // sample input matrix (16 rows, 4 cols), consists of 4 input vectors (grouping of 4 is used)
        -5, 9, -7, 4, // in case of affine layer this is the right operand of matrix mul
        5, -4, -7, 4, // in this sample the numbers are random and meaningless
        0, 7, 1, -7,
        1, 6, 7, 9,
        2, -4, 9, 8,
        -5, -1, 2, 9,
        -8, -8, 8, 1,
        -7, 2, -1, -1,
        -9, -5, -8, 5,
        0, -1, 3, 9,
        0, 8, 1, -2,
        -9, 8, 0, -7,
        -9, -8, -1, -4,
        -3, -7, -2, 3,
        -8, 0, 1, 3,
        -4, -6, -8, -2
    };

    int32_t biases[H] = {
        // sample bias vector, will get added to each of the four output vectors
        5, // in this sample the numbers are random and meaningless
        4,
        -2,
        5,
        -7,
        -5,
        4,
        -1
    };

    // Check number of available devices
    auto status = Gna2StatusSuccess;
    uint32_t deviceCount;
    status = Gna2DeviceGetCount(&deviceCount);
    CleanupOrError(status, GNA2_DISABLED, nullptr, GNA2_DISABLED, GNA2_DISABLED,
                   "main", "Gna2DeviceGetCount()");

    // [optional] Check version of first device
    auto deviceIndex = deviceCount - 1;
    auto version = Gna2DeviceVersionSoftwareEmulation;
    status = Gna2DeviceGetVersion(deviceIndex, &version);
    CleanupOrError(status, GNA2_DISABLED, nullptr, GNA2_DISABLED, GNA2_DISABLED,
                   "main", "Gna2DeviceGetCount()");
    if (Gna2DeviceVersionSoftwareEmulation == version)
    {
        printf("GNA Hardware Device not available, using Gna2DeviceVersionSoftwareEmulation.\n");
    }
    else
    {
        printf("GNA Hardware Device found: %d\n", static_cast<int32_t>(version));
    }

    // Open selected  device
    status = Gna2DeviceOpen(deviceIndex);
    CleanupOrError(status, deviceIndex, nullptr, GNA2_DISABLED, GNA2_DISABLED,
                   "main", "Gna2DeviceOpen()");

    /* Calculate model memory parameters for GnaAlloc. */
    int buf_size_weights = Gna2RoundUpTo64(sizeof(weights));
    // note that buffer alignment to 64-bytes is required by GNA HW
    int buf_size_inputs = Gna2RoundUpTo64(sizeof(inputs));
    int buf_size_biases = Gna2RoundUpTo64(sizeof(biases));
    int buf_size_outputs = Gna2RoundUpTo64(H * B * 4); // (4 out vectors, H elems in each one, 4-byte elems)

    auto rw_buffer_size = Gna2RoundUp(buf_size_inputs + buf_size_outputs, 0x1000);
    auto bytes_requested = rw_buffer_size + buf_size_weights + buf_size_biases;

    // Allocate GNA memory (obtains pinned memory shared with the device)
    uint32_t bytes_granted;
    void* memory = nullptr;
    status = Gna2MemoryAlloc(bytes_requested, &bytes_granted, &memory);
    CleanupOrError(status, deviceIndex, memory, GNA2_DISABLED, GNA2_DISABLED,
                   "main", "Gna2MemoryAlloc()");


    /* Prepare model memory layout. */
    auto model_memory = reinterpret_cast<uint8_t*>(memory);

    auto rw_buffers = model_memory;

    auto pinned_inputs = reinterpret_cast<int16_t*>(rw_buffers);
    memcpy_s(pinned_inputs, buf_size_inputs, inputs, sizeof(inputs)); // puts the inputs into the pinned memory
    rw_buffers += buf_size_inputs; // fast-forwards current pinned memory pointer to the next free block

    auto pinned_outputs = reinterpret_cast<int32_t*>(rw_buffers);
    rw_buffers += buf_size_outputs; // fast-forwards the current pinned memory pointer by the space needed for outputs

    model_memory += rw_buffer_size;
    auto weights_buffer = reinterpret_cast<int16_t*>(model_memory);
    memcpy_s(weights_buffer, buf_size_weights, weights, sizeof(weights)); // puts the weights into the pinned memory
    model_memory += buf_size_weights; // fast-forwards current pinned memory pointer to the next free block

    auto biases_buffer = reinterpret_cast<int32_t*>(model_memory);
    memcpy_s(biases_buffer, buf_size_biases, biases, sizeof(biases)); // puts the biases into the pinned memory
    model_memory += buf_size_biases; // fast-forwards current pinned memory pointer to the next free block

    /* Prepare neural network topology,
     * Single FullyConnectedAffine layer in this example. */

    /* Prepare and initialize FullyConnectedAffine operation operands with GNA API helpers */
    auto inputTensor = Gna2TensorInit2D(W, B, Gna2DataTypeInt16, pinned_inputs);
    auto outputTensor = Gna2TensorInit2D(H, B, Gna2DataTypeInt32, pinned_outputs);
    auto weightTensor = Gna2TensorInit2D(H, W, Gna2DataTypeInt16, weights_buffer);
    auto biasTensor = Gna2TensorInit1D(H, Gna2DataTypeInt32, biases_buffer);
    auto activationTensor = Gna2TensorInitDisabled();

    /* Create single FullyConnectedAffine operation (layer) */
    auto operation = Gna2Operation{};
    status = Gna2OperationInitFullyConnectedAffine(&operation, customAlloc,
                                                   &inputTensor, &outputTensor, &weightTensor, &biasTensor,
                                                   &activationTensor);
    CleanupOrError(status, deviceIndex, memory, GNA2_DISABLED, GNA2_DISABLED,
                   "main", "Gna2OperationInitFullyConnectedAffine()");

    /* Create data-flow model with single operation (layer) */
    Gna2Model model = {1, &operation};
    uint32_t modelId = GNA2_DISABLED;
    status = Gna2ModelCreate(deviceIndex, &model, &modelId);
    CleanupOrError(status, deviceIndex, memory, modelId, GNA2_DISABLED,
                   "main", "Gna2ModelCreate()");

    // Create request configuration used for queueing inference requests
    uint32_t configId = GNA2_DISABLED;
    status = Gna2RequestConfigCreate(modelId, &configId);
    CleanupOrError(status, deviceIndex, memory, modelId, configId,
                   "main", "Gna2RequestConfigCreate()");

    // Set model input data buffer, operation 0, operand 0, for this sample
    status = Gna2RequestConfigSetOperandBuffer(configId, 0, 0, pinned_inputs);
    CleanupOrError(status, deviceIndex, memory, modelId, configId,
                   "main", "Gna2RequestConfigSetOperandBuffer(0, 0)");

    // Set model output data buffer, operation 0, operand 1, for this sample
    status = Gna2RequestConfigSetOperandBuffer(configId, 0, 1, pinned_outputs);
    CleanupOrError(status, deviceIndex, memory, modelId, configId,
                   "main", "Gna2RequestConfigSetOperandBuffer(0, 1)");

    // [optional] Set acceleration mode automatic (software emulation used if no hardware detected)
    status = Gna2RequestConfigSetAccelerationMode(configId, Gna2AccelerationModeAuto);
    CleanupOrError(status, deviceIndex, memory, modelId, configId,
                   "main", "Gna2RequestConfigSetAccelerationMode(Gna2AccelerationModeAuto)");

    // Enqueue inference request (non-blocking call)
    uint32_t requestId; // this gets filled with the actual id later on
    status = Gna2RequestEnqueue(configId, &requestId);
    CleanupOrError(status, deviceIndex, memory, modelId, configId,
                   "main", "Gna2RequestEnqueue()");

    /**************************************************************************************************
     * Offload effect: other calculations can be done on CPU here, while model inference runs on GNA HW *
     **************************************************************************************************/

    // Wait for inference request completion (blocks until the results are ready)
    uint32_t timeout = 1000;
    status = Gna2RequestWait(requestId, timeout);
    CleanupOrError(status, deviceIndex, memory, modelId, configId,
                   "main", "Gna2RequestWait(1000)");

    // Reference output:
    // -177  -85   29   28
    //   96 -173   25  252
    // -160  274  157  -29
    //   48  -60  158  -29
    //   26   -2  -44 -251
    // -173  -70   -1 -323
    //   99  144   38  -63
    //   20   56 -103   10
    print_outputs(reinterpret_cast<int32_t*>(pinned_outputs), H, B);

    status = Gna2RequestConfigRelease(configId);
    CleanupOrError(status, deviceIndex, memory, modelId, GNA2_DISABLED,
                   "main", "Gna2RequestConfigRelease()");

    status = Gna2ModelRelease(modelId);
    CleanupOrError(status, deviceIndex, memory, GNA2_DISABLED, GNA2_DISABLED,
                   "main", "Gna2ModelRelease()");

    status = Gna2MemoryFree(memory);
    CleanupOrError(status, deviceIndex, nullptr, GNA2_DISABLED, GNA2_DISABLED,
                   "main", "Gna2MemoryFree()");

    status = Gna2DeviceClose(deviceIndex);
    CleanupOrError(status, GNA2_DISABLED, nullptr, GNA2_DISABLED, GNA2_DISABLED,
                   "main", "Gna2DeviceClose()");

    return 0;
}
catch (...)
{
    printf("Unhandled exception was thrown.\n");
    return -1;
}
