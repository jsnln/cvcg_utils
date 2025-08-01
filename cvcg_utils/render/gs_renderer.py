
import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .camera import UnifiedCamera, GSCamera
from .sh_utils import eval_sh

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def quat_from_axis_angle(npts, axis, angle):
    axis = torch.nn.functional.normalize(axis, dim=-1)
    cos_angleo2 = math.cos(angle/2)
    sin_angleo2 = math.sin(angle/2)

    quats = torch.zeros(npts, 4)
    quats[:, 0] = cos_angleo2
    quats[:, 1] = sin_angleo2 * axis[0]
    quats[:, 2] = sin_angleo2 * axis[1]
    quats[:, 3] = sin_angleo2 * axis[2]

    return quats


def render_gs(camera: GSCamera,
           xyz: torch.Tensor,
           opacity: torch.Tensor,
           scales: torch.Tensor,
           rotations: torch.Tensor,
           features: torch.Tensor,
           active_sh_degree: int,
           cov3D_precomp: torch.Tensor = None,
           override_color = None,
           compute_cov3D_python: bool = False,
           convert_SHs_python: bool = False,
           use_trained_exp=False,
           clip_value=False,
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    xyz:            [N, 3]
    opacity:        [N, 1]
    scales:         [N, 3], overridable by cov3D_precomp
    rotations:      [N, 3], overridable by cov3D_precomp
    cov3D_precomp:  [N, 6], stores only the lower diag. use `strip_symmetric` to obtain this)
    override_color: [N, 3]
    ...
    """

    assert features.shape[1] >= (active_sh_degree + 1) ** 2
    rasterizer = make_gs_rasterizer(camera, bg_color=torch.tensor([0., 0., 0.], device=xyz.device), active_sh_degree=active_sh_degree)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = xyz
    means2D = screenspace_points

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    # ????????????????

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if clip_value:
        rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        # "depth" : depth_image,
        }
    
    return out

def make_gs_rasterizer(camera: GSCamera, bg_color, scaling_modifier=1.0, active_sh_degree=3, debug=False, antialiasing=False):

    # FoVx, FoVy, tanfovx, tanfovy, world_view_transform, projection_matrix, full_proj_transform, camera_center = camera.to_3dgs_format()

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.H),
        image_width=int(camera.W),
        tanfovx=camera.tanfov_x,
        tanfovy=camera.tanfov_y,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform.clone().float().cuda(),
        projmatrix=camera.full_proj_transform.clone().float().cuda(),
        sh_degree=active_sh_degree,
        campos=camera.camera_center.clone().float().cuda(),
        prefiltered=False,
        debug=debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    return rasterizer