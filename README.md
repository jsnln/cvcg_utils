# CV & CG Utils

Easy-to-use and pytorch-compatible wrappers for various CV & CG tools:

- pytorch, pytorch3d
- libigl
- drtk
- nvdiffrecmc

### Usage & Installation

Do `pip install .` here to install the repo to your python package directory.

Note that this repo mainly provide APIs that calls other common packages. Imports are kept as independent as possible so that you only need to install needed modules.

### 1. Mesh utils in `mesh`

Dependencies: `libigl`, `plyfile`, `pygltflib`, `opencv-python`, `torch`, `scipy`

### 2. DRTK rendering utils in `render`

Dependencies: `torch`, `drtk`
