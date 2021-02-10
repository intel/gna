/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

Build instruction for samples.

CMake (ver. at least 3.10 is required).
For additional information about how project is built consult CMakeLists.txt file.

1. Generate projects with CMake:
	Example command:
----------------------------------------------------------------------------
		Microsoft* Windows* specific:
		cmake -G "Visual Studio 15 2017 Win64" .
----------------------------------------------------------------------------
		LINUX specific:
		cmake .
----------------------------------------------------------------------------
2. Build project:
   cmake --build .
3. Executable should be under src/sample01/Debug directory.

*Other names and brands may be claimed as the property of others.
