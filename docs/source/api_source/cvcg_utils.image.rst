cvcg\_utils.image
=========================

The subpackage contains tools for

1. image IO;
2. image processing

.. contents::

cvcg\_utils.image.image\_io
----------------------------------

In ``cvcg_utils.image.image_io``, each read/write method is also marked by the image format (e.g., whether it's RGB or RGBA) and dtype (e.g., ``np.uint8`` or ``np.float32``). In research code, not knowing the exact data type (which usually happens if no data type check is done) can potentially cause very obscure bugs. Our choice is to sacrifice flexibility for readability and stability.

.. note::

   All image IO methods use the ``cv2`` backend. The BGR-RGB conversion is handled by our wrappers. 

.. automodule:: cvcg_utils.image.image_io
   :members:
   :show-inheritance:
   :undoc-members:

cvcg\_utils.image.image\_proc
------------------------------------

.. automodule:: cvcg_utils.image.image_proc
   :members:
   :show-inheritance:
   :undoc-members:

.. Module contents
.. ---------------

.. .. automodule:: cvcg_utils.image
..    :members:
..    :show-inheritance:
..    :undoc-members:
