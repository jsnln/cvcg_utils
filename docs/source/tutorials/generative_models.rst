Generative Models
=================

.. highlight:: python

Stochastic Differential Equation
--------------------------------

Since almost all generative models can be unified within a SDE (Stochastic Differential Equation), we start with an introduction to SDE.

Let :math:`p_0` be the data distribution in :math:`\mathbb R^d` and let :math:`p_T` be a prior distribution (usually but not necessarily a standard Gaussian) also over :math:`\mathbb R^d`. Our goal is to construct a continuous stochastic process :math:`\{x(t)\}` indexed by a continous time variable :math:`t\in[0,T]`, such that :math:`x(0)\sim p_0` and :math:`x(T)\sim p_T`. Additionally, we assume :math:`x(t)` is characterized by the following Itô SDE:

.. math::

    {\rm d}x=f(x,t){\rm d}t+g(t){\rm d}w.

Here, :math:`w` is the standard Wiener process (Brownian motion), i.e., it adds Gaussian noises to :math:`x(t)` when integrated over time :math:`t`. :math:`f(\cdot,t):\mathbb R^d\to\mathbb R^d` maps the current :math:`x(t)` to a velocity vector, usually called the *drift* coefficient of :math:`x(t)`. :math:`g(t)\in\mathbb R` determines the scale of the Brownian motion added at time :math:`t`, called the *diffusion* coefficient of :math:`x(t)`. Intuitively, this gradually adds Gaussian noise to :math:`x(t)`, so it is also called a forward diffusion process.

.. note::

    A standard Wiener process is defined as a stochastic process :math:`\{w_t\}_{t\geq0}` that satisfies:

    1. :math:`w_0=0`.
    2. :math:`w_t` is continous in :math:`t` almost everywhere.
    3. The increments :math:`w_{t+s}-w_s` observes :math:`\mathcal N(0,tI)` for any :math:`s,t>0`, and the increments are stationary and independent (see e.g. `these lecture notes <UchicagoBrownian_>`__).

    We can derive from (3) that if we let :math:`z=w_{t+s}-w_s`, then :math:`z\sim\mathcal N(0,tI)`. Thus, :math:`{\rm d}w=\sqrt{{\rm d}t}z` where :math:`z\sim\mathcal N(0,tI)`.

.. _UchicagoBrownian: https://galton.uchicago.edu/~lalley/Courses/313/BrownianMotionCurrent.pdf


According to [Anderson1982]_, the reverse process is given by

.. math::

    {\rm d}x=(f(x,t)-g(t)^2\nabla_x\log p_t(x)){\rm d}t+g(t){\rm d}\bar w,

where :math:`p_t(x)` is the probability density of :math:`x(t)`, and :math:`\bar w` is a Wiener process with time going backwards. Note that the reverse SDE should be integrated with :math:`t` going back from :math:`T` to :math:`0`, i.e., :math:`{\rm d}t<0`.

In both forward and reverse processes, :math:`f(x,t)` and :math:`g(t)` are given by the specific models we choose, and they can be considered as known function. In order to sample from the reverse process, we additionally need to know :math:`\nabla_x\log p_t(x)`, known as the *score function* of the distribution :math:`p_t(x)`.



Estimating Score Functions
--------------------------

Note that

.. math::

    \nabla_x\log p_t(x)=\nabla_{x(t)}\log(p_{0,t}(x(t)|x(0))p_0(x(0)))=\nabla_{x(t)}\log p_{0,t}(x(t)|x(0)),

where :math:`p_{s,t}(x(t)|x(s))` is called the *transition kernel* from :math:`s` to :math:`t` (:math:`s<t`). To estimate the score function, we can start from sampling :math:`x(0)\sim p_0` and some :math:`t\in[0,T]`, diffuse it to get :math:`x(t)\sim p_t`, and then compute :math:`\nabla_{x(t)}\log p_{t}(x(t))=\nabla_{x(t)}\log p_{0,t}(x(t)|x(0))`. Here, :math:`p_{0,t}(x(t)|x(0))` is known and depends on :math:`f(x,t)` and :math:`g(t)` (but computing it is another matter). Finally, a neural network :math:`s_\theta(x(t),t)` parameterized by :math:`\theta` is trained to estimate the score function by minimizing:

.. math::

    \min_\theta\mathbb E_{t,x(0),x(t)|x(0)}\left[\|s_\theta(x(t),t)-\nabla_{x(t)}\log p_{0,t}(x(t)|x(0))\|^2\right].

Supposably, :math:`p_{0,t}(x(t)|x(0))` is Gaussian when :math:`f(x,t)` is affine in :math:`x` (haven't checked this myself).

.. note::

    The sampling of :math:`t` is equivalent to reweighting the loss according to different :math:`t`, and is usually also equivalent to weighting brought by differnet schedules (choices of :math:`f` and :math:`g`).



Forward Processes
-----------------

Score Matching with Langevin Dynamics (SMLD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For SMLD, the forward process is given by

.. math::

    p_{\sigma_t}(x(t)|x(0))=\mathcal N(x(0),\sigma_t^2I)
    
    \iff x(t)=x(0)+\sigma_tz,\ z\sim\mathcal N(0,I).

Here, :math:`\sigma_t` is monotonically increasing in :math:`t`. Consider discrete time steps :math:`0=t_0<t_1<t_2<\cdots<t_N` (for simplicity we define :math:`\sigma_{t_0}=0`). Then the discrete forward process is

.. math::

    x_{t_i}=x_{t_0}+\sigma_{t_i}z,\ z\sim\mathcal N(0,I).

The corresponding incremental version is

.. math::

    x_{t_i}=x_{t_{i-1}}+\sqrt{\sigma_{t_i}^2-\sigma_{t_{i-1}}^2}z,\ z\sim\mathcal N(0,I).

.. todo::

    I have not idea how to convert between incremental update rules and the one-shot rule.

Then we have

.. math::

    \frac{x_{t_i}-x_{t_{i-1}}}{t_i-t_{i-1}}=\sqrt{\frac{\sigma_{t_i}^2-\sigma_{t_{i-1}}^2}{t_i-t_{i-1}}}\frac{z}{\sqrt{t_i-t_{i-1}}},\ z\sim\mathcal N(0,I).

Taking the limit :math:`t_i\to t_{i-1}` and noting that :math:`{\rm d}w=\sqrt{{\rm d}t}z` we get

.. math::

    {\rm d}x=\sqrt{\frac{{\rm d}\sigma^2(t)}{{\rm d}t}}{\rm d}w.

Denoising Diffusion Probabilistic Models (DDPM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For DDPM, the forward process is defined as

.. math::

    p_{\alpha_t}(x(t)|x(0))=\mathcal N(\sqrt{\alpha_t}x(0),(1-\alpha_t)I)
    
    \iff x(t)=\sqrt{\alpha_t}x(0)+\sqrt{1-\alpha_t}z,\ z\sim\mathcal N(0,I).

Again, let :math:`0=t_0<t_1<t_2<\cdots<t_N` and :math:`\alpha_0=1`. Then the discrete form is

.. math::

    x_{t_i}=\sqrt{\alpha_{t_i}}x_{t_0}+\sqrt{1-\alpha_{t_i}}z,\ z\sim\mathcal N(0,I).

The equivalent incremental form is

.. math::

    x_{t_i}=\sqrt{1-\beta_{t_i}}x_{t_{i-1}}+\sqrt{\beta_{t_i}}z,\ z\sim\mathcal N(0,I),

where :math:`\beta_{t_i}\in(0,1)` are noise scales such that :math:`\alpha_{t_i}=\prod_{j=1}^i(1-\beta_{t_j})`. Similar to SMLD, we have

.. math::

    \frac{x_{t_i}-x_{t_{i-1}}}{t_i-t_{i-1}}\approx-\frac{\beta_{t_i}}{2(t_i-t_{i-1})}x_{t_{i-1}}+\sqrt{\frac{\beta_{t_i}}{t_i-t_{i_1}}}\frac{z}{\sqrt{t_i-t_{i_1}}},\ z\sim\mathcal N(0,I),
.. rubric:: References

Note that as :math:`t_i\to t_{i-1}`, we also have :math:`\beta_{t_i}\to0`. So we need to define :math:`\beta(t_{i-1})=\lim_{t_i\to t_{i-1}}\beta_{t_i}/(t_i-t_{i-1})`. In this way, the continuous version of DDPM is

.. math::

    {\rm d}x=-\frac{\beta(t)}{2}x{\rm d}t+\sqrt{\beta(t)}{\rm d}w.

Flow Matching (FM)
^^^^^^^^^^^^^^^^^^

FM itself is a continous model defined by

.. math::

    x(t)=(1-t)x(0)+tz,\ z\sim\mathcal N(0,I),

which gives

.. math::

    p_t(x(t)|x(0))=\mathcal N((1-t)x(0),t^2I).


.. [Anderson1982] Brian D. O. Anderson. *Reverse-time diffusion equation models*. Stochastic Processes and their Applications, 12(3):313-326, May 1982.

