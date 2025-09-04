CV & CG Utils
=============

A repository hosting easy-to-use and pytorch-compatible wrappers for various CV & CG tools, including not not limited to
PyTorch3D, DRTK, libigl, nvdiffrecmc.

**Features**

  1. **Code-only** You don't need to compile anything to install this package itself.
  2. **Independent imports and minimal dependency** While we use a lot of external packages, imports are kept as independent as possible. Dependencies can be installed on a need-to-use basis.
  3. **Unified and simplifed API** We try the best to provide simple yet unified data structures and API for different backend tools.

**However**

  1. One drawback of using independent imports is that you have to imports different subpackages one-by-one. This might get tedious.
  2. The package is mostly research-oriented, which means we choose readability and extendability over runtime efficiency. The implementations are not always optimal.


Getting Started
---------------

``pip install .``

Note that this repo mainly provide APIs that calls other common packages. Imports are kept as independent as possible so that you only need to install needed modules.

Mesh utils in ``mesh``
^^^^^^^^^^^^^^^^^^^^^^

Dependencies: ``libigl, plyfile, pygltflib, opencv-python, torch, scipy``

DRTK rendering utils in ``render``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dependencies: ``torch, drtk``
