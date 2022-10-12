/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

/**************************************************************************//**
 @file gna2-model-export-api.h
 @brief Gaussian and Neural Accelerator (GNA) 3.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_MODEL_API
 @{
 *****************************************************************************

 @addtogroup GNA2_MODEL_EXPORT_API Model Export API

 API for exporting GNA model for embedded devices.

 @{
 *****************************************************************************/

#ifndef __GNA2_MODEL_EXPORT_API_H
#define __GNA2_MODEL_EXPORT_API_H

#include "gna2-common-api.h"

#if !defined(_WIN32)
#include <cassert>
#endif
#include <cstdint>

/**
 Creates configuration for model exporting.

 Export configuration allows to configure all the parameters necessary
 to export components of one or more models.
 Use Gna2ModelExportConfigSet*() functions to configure parameters. Parameters
 can be modified/overridden for existing configuration to export model
 with modified properties.

 @warning
    User is responsible for releasing allocated memory buffers.

 @param userAllocator User provided memory allocator.
 @param [out] exportConfigId Identifier of created export configuration.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelExportConfigCreate(
    Gna2UserAllocator userAllocator,
    uint32_t * exportConfigId);

/**
 Releases export configuration and all its resources.

 @param exportConfigId Identifier of export configuration to release.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelExportConfigRelease(
    uint32_t exportConfigId);

/**
 Sets source model(s) to export.

 - Model will be validated against provided device.
 - Model(s) should be created through standard API Gna2ModelCreate() function.

 @param exportConfigId Identifier of export configuration to set.
 @param sourceDeviceIndex Id of the device on which the exported model was created.
 @param sourceModelId Id of the source model, created previously with Gna2ModelCreate() function.
 @return Status of the operation.
 @retval Gna2StatusIdentifierInvalid when sourceDeviceIndex is not matching the one
         retrieved from ::Gna2DeviceCreateForExport() or when source device has no model
         with sourceModelId.

 */
GNA2_API enum Gna2Status Gna2ModelExportConfigSetSource(
    uint32_t exportConfigId,
    uint32_t sourceDeviceIndex,
    uint32_t sourceModelId);

/**
 @deprecated Currently targetDeviceVersion is already set when using ::Gna2ModelExportConfigSetSource().
 This function is no longer necessary and will return success when given targetDeviceVersion
 is compatible with targetDeviceVersion provided in ::Gna2DeviceCreateForExport,
 Gna2StatusDeviceVersionInvalid otherwise.

 Sets version of the device that exported model will be used with.

 - Model will be validated against provided target device.

 @param exportConfigId Identifier of export configuration to set.
 @param targetDeviceVersion Device on which model will be used.
 @return Status of the operation.
 @retval Gna2StatusDeviceVersionInvalid when targetDeviceVersion is not compatible
         with targetDeviceVersion provided in ::Gna2DeviceCreateForExport.
 */
GNA2_API enum Gna2Status Gna2ModelExportConfigSetTarget(
    uint32_t exportConfigId,
    enum Gna2DeviceVersion targetDeviceVersion);

/**
 Determines the type of the component to export.
 */
enum Gna2ModelExportComponent
{
    /**
     Hardware layer descriptors will be exported.
     */
    Gna2ModelExportComponentLayerDescriptors = GNA2_DEFAULT,

    /**
     Header describing layer descriptors will be exported.
     */
    Gna2ModelExportComponentLayerDescriptorHeader = 1,

    /**
     Hardware layer descriptors and model data in legacy SueCreek format will be exported.

     @note:
     To support RW and RO data separation 2 allocations
     for RW and RO memory (in order) are required.
     */
    Gna2ModelExportComponentLegacySueCreekDump = 2,

    /**
     Header describing layer descriptors in legacy SueCreek format will be exported.
     @note:
     To support RW and RO data separation 2 allocations
     for RW and RO memory (in order) are required.
    */
    Gna2ModelExportComponentLegacySueCreekHeader = 3,

    Gna2ModelExportComponentReadOnlyDump = 4,

    Gna2ModelExportComponentScratchDump = 6,
    Gna2ModelExportComponentStateDump = 7,

    Gna2ModelExportComponentInputDump = 11,
    Gna2ModelExportComponentOutputDump = 12,

    Gna2ModelExportComponentExternalBufferInputDump = 21,
    Gna2ModelExportComponentExternalBufferOutputDump = 22,
};

/**
 Type of exported memory buffer tag.

 Used to tag memory buffer and assign special properties or BAR for exported memory buffer.
 @see Gna2MemorySetTag()
 */
enum Gna2MemoryTag
{
    Gna2MemoryTagReadWrite = 0x0100,
    Gna2MemoryTagInput = 0x0200,
    Gna2MemoryTagOutput = 0x0400,
    Gna2MemoryTagReadOnly = 0x0800,
    Gna2MemoryTagExternalBufferInput = 0x1000,
    Gna2MemoryTagExternalBufferOutput = 0x2000,
    Gna2MemoryTagScratch = 0x4000,
    Gna2MemoryTagState = 0x8000,
};

/**
 Exports the model(s) component.

 All exported model components are saved into memory allocated on user side by userAllocator.

 @warning
    User is responsible for releasing allocated memory buffers (exportBuffer).

 @param exportConfigId Identifier of export configuration used.
 @param componentType What component should be exported.
 @param [out] exportBuffer Memory allocated by userAllocator with exported layer descriptors.
 @param [out] exportBufferSize The size of exportBuffer in bytes.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelExport(
    uint32_t exportConfigId,
    enum Gna2ModelExportComponent componentType,
    void ** exportBuffer,
    uint32_t * exportBufferSize);

#endif // __GNA2_MODEL_EXPORT_API_H

/**
 @}
 @}
 @}
 */
