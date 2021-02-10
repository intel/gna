@echo off

::@copyright (C) 2020-2021 Intel Corporation
::SPDX-License-Identifier: LGPL-2.1-or-later

echo Building and publish  GNA API Doxygen documentation

echo PROJECT_NUMBER=%GNA_LIBRARY_VERSION% >> Doxyfile || exit /b 666
doxygen || exit /b 666
