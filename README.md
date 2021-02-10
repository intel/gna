# GNA - Gaussian & Neural Accelerator Library repository
[![LGPL-2.1-or-later](https://img.shields.io/badge/license-lgpl_2.1_or_later-green.svg)](LICENSE)

Intel® Gaussian & Neural Accelerator is a low-power neural coprocessor for continuous inference at the edge.

When power and performance are critical, the Intel® Gaussian & Neural Accelerator (Intel® GNA) provides power-efficient, always-on support. Intel® GNA is designed to deliver AI speech and audio applications such as neural noise cancellation, while simultaneously freeing up CPU resources for overall system performance and responsiveness.

GNA library provides an API to run inference on Intel® GNA hardware, as well as in the software execution mode on CPU.

GNA library is also a part of [OpenVINO™](https://github.com/openvinotoolkit/openvino).

Intel® GNA hardware requires a driver to be installed on the system. For Windows\* please see:
[Intel® Drivers \& Software](https://downloadcenter.intel.com/download/30139/Intel-GNA-Scoring-Accelerator-Driver-for-Intel-NUC11TN?wapkw=gna) or Windows\* Update.


## Repository components:
* GNA library
  * kernels (Software emulation kernels)
    * GMM (Gaussian Mixture Models kernels)
    * XNN (Neural Network kernels)
  * gna-api (core library and API)
* samples (minimalistic usage example)

## License
GNA library is licensed under [GNU Lesser General Public License v2.1 or later](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## Resources:
* [Introducing the GNA Plugin](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_GNA.html)
* [OpenVINO™ Toolkit Wiki](https://github.com/openvinotoolkit/openvino/wiki)
* [OpenVINO™ Toolkit HomePage](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

## Support
Please report questions, issues and suggestions using:

* [GitHub* Issues](https://github.com/intel/gna/issues)
* [OpenVINO™ Toolkit Forum](https://software.intel.com/en-us/forums/computer-vision)

---
\* Other names and brands may be claimed as the property of others.