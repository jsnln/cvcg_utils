
import torch
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from diff_gaussian_rasterization_debug import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_depth_trunc import GaussianRasterizationSettings, GaussianRasterizer
from .camera import UnifiedCamera
from .sh_utils import eval_sh

def render_gs(camera,
           xyz: torch.Tensor,
           opacity: torch.Tensor,
           scales: torch.Tensor,
           rotations: torch.Tensor,
           cov3D_precomp: torch.Tensor,
           override_color = None,
           compute_cov3D_python: bool = False,
           convert_SHs_python: bool = False,
           separate_sh = False,
           use_trained_exp=False,
           clip_value=False,
           termi_depth_img=None
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if termi_depth_img is None:
        zfar = 100
        termi_depth_img = zfar * torch.ones(camera.H, camera.W, device=xyz.device)

    rasterizer = make_gs_rasterizer(camera, bg_color=torch.tensor([0., 0., 0.], device=xyz.device))
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = xyz
    means2D = screenspace_points
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    # cov3D_precomp = None

    # if compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    assert override_color is not None
    if override_color is None:
        if convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            termi_depth_img = termi_depth_img)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            termi_depth_img = termi_depth_img)
        
    # Apply exposure to rendered image (training only)
    # ????????????????
    assert not use_trained_exp
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if clip_value:
        rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out

def make_gs_rasterizer(camera: UnifiedCamera, bg_color, scaling_modifier=1.0, active_sh_degree=0, debug=False, antialiasing=False):

    FoVx, FoVy, tanfovx, tanfovy, world_view_transform, projection_matrix, full_proj_transform, camera_center = camera.to_3dgs_format()

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.H),
        image_width=int(camera.W),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=torch.from_numpy(world_view_transform).float().cuda(),
        projmatrix=torch.from_numpy(full_proj_transform).float().cuda(),
        sh_degree=active_sh_degree,
        campos=torch.from_numpy(camera_center).float().cuda(),
        prefiltered=False,
        debug=debug,
        antialiasing=antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    return rasterizer