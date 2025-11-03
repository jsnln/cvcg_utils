Computer Vision
===============

.. highlight:: python

Camera Parametrization (OpenCV)
-------------------------------

Cameras are mostly commonly denoted by their intrinsics :math:`K` and extrinsics :math:`R, t`. In the OpenCV convention, :math:`K` is in pixel units, and :math:`R,t` denote world-to-camera transformation.

Camera Parametrization (Plücker)
--------------------------------

Another commonly used parametrization is called **Plücker rays**, where the pixel is associated with the ray shooting from the camera center :math:`p` to the observed point. The ray at :math:`(u,v)` is parametrized by :math:`d_{u,v}\in\mathbb{R}^3` and :math:`m_{u,v}\in\mathbb{R}^3`. Here, :math:`d_{u,v}` has unit length and denotes the direction of the ray, and :math:`m_{u,v}=p\times d_{u,v}` is the momentum.

.. note::

    One can show that :math:`p` can be replaced by any point on the ray and :math:`m_{u,v}` stays in variant. Moreover, :math:`(d_{u,v}, m_{u,v})` uniquely determines a ray with direction.

For each pixel :math:`(u,v)` of a camera with parameters :math:`K,R,t`, we can compute its Plücker ray parametrization as

.. math::

    p & = -R^Tt\\
    \tilde d_{u,v} & = R^TK^{-1}\left[\begin{matrix}u\\v\\1\end{matrix}\right]\\
    d_{u,v} & = \frac{\tilde d_{u,v}}{\|\tilde d_{u,v}\|}\\
    m_{u,v} & = p\times d_{u,v}

On the other hand, to recover :math:`K,R,t` from Plücker rays takes some more work. First, we solve the camera center as the intersection of all rays (in the least square sense):

.. math::

    p = \mathop{\mathrm{argmin}}_{p}\sum_{u,v}\|p\times d_{u,v}-m_{u,v}\|^2

After obtaining :math:`p`, we note that :math:`K, R` satisfy (:math:`\sim` denotes equal up to scaling)

.. math::

    Rd_{u,v}\sim K^{-1}\left[\begin{matrix}u\\v\\1\end{matrix}\right]

Or equivalently:

.. math::

    \exists\alpha_{u,v}, (KR)d_{u,v}=\alpha_{u,v}\left[\begin{matrix}u\\v\\1\end{matrix}\right]

Now let us define :math:`P=KR` (the rotational of part of the overall projection matrix), :math:`h_{u,v}=[u,v,1]^T` (the homogeneous pixel coordinates of :math:`(u,v)`). Then we have

.. math::

    \exists\alpha_{u,v}, Pd_{u,v}=\alpha_{u,v}h_{u,v}

Using DLT removes the unknown constant :math:`\alpha_{u,v}`

.. math::

    0=h_{u,v}\times (Pd_{u,v})=[h_{u,v}]_\times Pd_{u,v}

This is a linear equation on :math:`P`, to solve it, we need to rewrite the linear operator :math:`[h_{u,v}]_\times Pd_{u,v}` as a matrix-vector product :math:`B(P)=B\ \mathrm{Flatten}(P)`. However, using ``scipy.sparse.linalg.svds`` and ``scipy.sparse.linalg.LinearOperator`` we can use an operator to represent :math:`B` and :math:`B^T`, as follows:

.. math::

         B(P)           & = [h_{u,v}]_\times Pd_{u,v} \in\mathbb{R}^3,&\ \textrm{for}\ P\in\mathbb{R}^{3\times3}\\
    \iff B(P)_i         & = \sum_{j,k}([h_{u,v}]_\times)_{ij}(d_{u,v})_{k}P_{jk}\\
    \iff B^T(r)_{jk}    & = \sum_{i}([h_{u,v}]_\times)_{ij}r_i\\
                        & = \sum_{i}([h_{u,v}]_\times)_{ij}r_i(d_{u,v})_{k}\\
                        & = ([h_{u,v}]_\times^Tr)_j(d_{u,v})_{k}\\
                        & = (-h_{u,v}\times r)_j(d_{u,v})_{k}\\
    \iff B^T(r)         & = (-h_{u,v}\times r)(d_{u,v})^T \in\mathbb{R}^{3\times3},&\ \textrm{for}\ r\in\mathbb{R}^3

Finally, :math:`P` is solved as the singular vector of :math:`B` with the smallest singular value, with :math:`\det(P)>0`. And then we do an RQ decomposition (bad naming in this case, our :math:`R` is actually the :math:`Q`, and our :math:`K` is the :math:`R`):

.. math::

    P=\tilde K\tilde R

.. note::

    It is important that you choose :math:`P` to have :math:`\det(P)>0`. Furthermore, using the ``arpack`` solver will ensure :math:`\det(\tilde R)>0` because it uses 2 Householder reflections, in which case we also have :math:`\det(\tilde K)>0`.

Here :math:`\tilde K` is upper triangular and :math:`\tilde R` is orthogonal. Note that they are not the final intrinsics and extrinsics yet due to scale and orientation ambiguity. Note the following (:math:`\tilde K_{12}` may be a non-zero but very small number, we ignore it here):

.. math::

    \tilde K\tilde R = \left[\begin{matrix}\tilde K_{11} & 0 & \tilde K_{13}\\0 & \tilde K_{22} & \tilde K_{23}\\0 & 0 & \tilde K_{33}\end{matrix}\right]\left[\begin{matrix}\tilde r_1^T\\\tilde r_2^T\\\tilde r_3^T\end{matrix}\right]=\left[\begin{matrix}\tilde K_{11}\tilde r_1^T+\tilde K_{13}\tilde r_3^T\\\tilde K_{22}\tilde r_2^T+\tilde K_{23}\tilde r_3^T\\\tilde K_{33}\tilde r_3^T\end{matrix}\right]

So, reversing a sign for some column in :math:`\tilde K` is equivalent to reversing a sign for some row in :math:`\tilde R`. We can now correct the orientation using the following procedure:

1. Check if :math:`\tilde K_{11}<0`. If so, reverse the signs of the first column of :math:`\tilde K` and :math:`\tilde r_1^T`.

2. Check if :math:`\tilde K_{22}<0`. If so, reverse the signs of the second column of :math:`\tilde K` and :math:`\tilde r_2^T`.

3. Check if :math:`\tilde K_{33}<0`. If so, reverse the signs of the third column of :math:`\tilde K` and :math:`\tilde r_3^T`.

Assuming :math:`\tilde K` and :math:`\tilde R` now have rectified orientation, we finally set


.. math::

    K=\tilde K/\tilde K_{22},\quad R=\tilde R,\quad t=-Rp.