from typing import Self, List
import cv2
import math
import numpy as np
import torch

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2*focal))
    

def assert_ints(*args):
    for arg in args:
        assert isinstance(arg, int)

def assert_floats(*args):
    for arg in args:
        assert isinstance(arg, float)

def get_projection_matrix_3dgs(znear, zfar, fov_x=None, fov_y=None, K=None, img_h=None, img_w=None, array_package = np, device = None):
    if K is None:
        tanHalffov_y = math.tan((fov_y / 2))
        tanHalffov_x = math.tan((fov_x / 2))
        top = tanHalffov_y * znear
        bottom = -top
        right = tanHalffov_x * znear
        left = -right
    else:
        near_fx = znear / K[0, 0]
        near_fy = znear / K[1, 1]

        left = - (img_w - K[0, 2]) * near_fx
        right = K[0, 2] * near_fx
        bottom = (K[1, 2] - img_h) * near_fy
        top = K[1, 2] * near_fy

    P = array_package.zeros((4, 4), device=device)


    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def opencv_extrinsics_to_opengl_world2view(R, T):
    # OpenCV extrinsics to OpenGL world2view
    w2c_opencv = np.eye(4)
    w2c_opencv[:3, :3] = R
    w2c_opencv[:3, 3] = T

    c2w_opencv = np.linalg.inv(w2c_opencv)
    c2w_opengl = c2w_opencv.copy()
    c2w_opengl[:, 1] *= -1
    c2w_opengl[:, 2] *= -1
    w2c_opengl = np.linalg.inv(c2w_opengl)

    return w2c_opengl

def opencv_intrinsics_to_opengl_projection(znear, zfar, K=None, img_h=None, img_w=None):
    P = np.zeros((4, 4))
    P[0, 0] = 2.0 * K[0,0] / img_w
    P[1, 1] = 2.0 * K[1,1] / img_h
    P[0, 2] = (img_w - 2 * K[0, 2]) / (img_w)   # NOTE there is a minus sign compared to the Gaussian one!
    P[1, 2] = (2 * K[1, 2] - img_h) / (img_h)
    P[3, 2] = -1.0
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * (zfar * znear) / (zfar - znear)
    return P

class DRTKCamera(torch.nn.Module):
    def __init__(self, proj_mat, w2c_mat, H, W):
        """
        proj and w2c follows OpenGL convention
        """
        super().__init__()
        self.proj_mat = torch.nn.Buffer(torch.from_numpy(proj_mat))  # [H, W]
        self.w2c_mat = torch.nn.Buffer(torch.from_numpy(w2c_mat))  # [H, W]
        self.H = H
        self.W = W

    def proj_points_to_camera(self, pts: torch.Tensor):
        """
        pts: [B, N, 3]

        out: camera space coordinates
        """
        assert len(pts.shape) == 3
        assert pts.shape[2] == 3

        # full projection
        pts_wld_homog = torch.nn.functional.pad(pts, (0,1), mode='constant', value=1)
        pts_cam_homog = torch.einsum('yx,bnx->bny', self.w2c_mat, pts_wld_homog)
        
        pts_cam_x, pts_cam_y, pts_cam_z, _ = pts_cam_homog.unbind(dim=-1)
        pts_cam = torch.stack([pts_cam_x, pts_cam_y, pts_cam_z], dim=-1)

        return pts_cam

    
    def unproj_depth_image(self, depth_img: torch.Tensor):
        """
        depth_img: [B, H, W]

        out_img: [B, 3, H, W]
        """
        assert len(depth_img.shape) == 3

        batch_size = depth_img.shape[0]

        # OpenGL convention NDC coords
        clip_x_axis = torch.linspace(-1., 1., self.W, dtype=depth_img.dtype, device=depth_img.device)
        clip_y_axis = torch.linspace(1., -1., self.H, dtype=depth_img.dtype, device=depth_img.device)

        clip_xy_img = torch.stack([clip_x_axis[None, :].expand(self.H, -1),
                                   clip_y_axis[:, None].expand(-1, self.W)],
                                   dim=0)[None].expand(batch_size, -1, -1, -1)
        
        clip_x_img, clip_y_img = clip_xy_img.unbind(dim=1)   # [B, H, W]

        cam_z_img = - depth_img
        cam_x_img = ((-clip_x_img) - self.proj_mat[0,2]) * cam_z_img / (self.proj_mat[0,0])
        cam_y_img = ((-clip_y_img) - self.proj_mat[1,2]) * cam_z_img / (self.proj_mat[1,1])

        cam_xyz_img = torch.stack([cam_x_img, cam_y_img, cam_z_img], dim=1)
        
        world_xyz_img = torch.einsum('ji,bjhw->bihw', self.w2c_mat[:3, :3], cam_xyz_img - self.w2c_mat[:3, 3][None, :, None, None])
        
        return cam_xyz_img, world_xyz_img



    def proj_points_to_drtk_screen(self, pts: torch.Tensor, detach_z: bool):
        """
        pts: [B, N, 3]

        out: DRTK screen space coordinates, (-0.5, -0.5) to (W-0.5, H-0.5)
        """
        assert len(pts.shape) == 3
        assert pts.shape[2] == 3

        # full projection
        pts_wld_homog = torch.nn.functional.pad(pts, (0,1), mode='constant', value=1)
        pts_cam_homog = torch.einsum('yx,bnx->bny', self.w2c_mat, pts_wld_homog)
        
        if not detach_z:
            pts_cam_x, pts_cam_y, pts_cam_z, _ = pts_cam_homog.unbind(dim=-1)
            pts_clip_homog = torch.einsum('yx,bnx->bny', self.proj_mat,  pts_cam_homog)
            pts_clip = pts_clip_homog[..., :3] / pts_clip_homog[..., [3]]
        else:
            pts_cam_x, pts_cam_y, pts_cam_z, _ = pts_cam_homog.unbind(dim=-1)
            pts_cam_z_detached = pts_cam_z.detach()
            pts_clip_x = -(self.proj_mat[0,0] * pts_cam_x / pts_cam_z_detached + self.proj_mat[0,2])
            pts_clip_y = -(self.proj_mat[1,1] * pts_cam_y / pts_cam_z_detached + self.proj_mat[1,2])
            pts_clip_z = -(self.proj_mat[2,2] + self.proj_mat[2,3] / pts_cam_z_detached)
            pts_clip = torch.stack([pts_clip_x, pts_clip_y, pts_clip_z], dim=-1)

        # clip space (OpenGL) to screen space (DRTK)
        pts_x_screen = (pts_clip[..., 0] + 1) / 2 * self.W - 0.5
        pts_y_screen = (1 - pts_clip[..., 1]) / 2 * self.H - 0.5
        # pts_z_screen = pts_clip[..., 2]   # TODO, DRTK document says this should be the actual z coords, change later
        pts_z_screen = -pts_cam_z

        pts_screen = torch.stack([pts_x_screen,
                                  pts_y_screen,
                                  pts_z_screen], dim=-1)
        
        return pts_screen
    
    def transform_vectors_to_camera_frame(self, vecs: torch.Tensor):
        """
        pts: [B, N, 3]
        """
        assert len(vecs.shape) == 3
        assert vecs.shape[2] == 3

        return torch.einsum('yx,bnx->bny', self.w2c_mat[:3, :3], vecs)
         
    def proj_points_to_grid_sample(self, pts: torch.Tensor, detach_z: bool):
        """
        pts: [B, N, 3]

        out: torch.nn.functional.grid_sample compatible normalized coordinates, (-1, -1) to (1, 1), use align_corners=False
        """
        assert len(pts.shape) == 3
        assert pts.shape[2] == 3

        # full projection
        pts_wld_homog = torch.nn.functional.pad(pts, (0,1), mode='constant', value=1)
        pts_cam_homog = torch.einsum('yx,bnx->bny', self.w2c_mat, pts_wld_homog)
        
        if not detach_z:
            pts_clip_homog = torch.einsum('yx,bnx->bny', self.proj_mat,  pts_cam_homog)
            pts_clip = pts_clip_homog[..., :3] / pts_clip_homog[..., [3]]
        else:
            pts_cam_x, pts_cam_y, pts_cam_z, _ = pts_cam_homog.unbind(dim=-1)
            pts_cam_z_detached = pts_cam_z.detach()
            pts_clip_x = -(self.proj_mat[0,0] * pts_cam_x / pts_cam_z_detached + self.proj_mat[0,2])
            pts_clip_y = -(self.proj_mat[1,1] * pts_cam_y / pts_cam_z_detached + self.proj_mat[1,2])
            pts_clip_z = -(self.proj_mat[2,2] + self.proj_mat[2,3] / pts_cam_z_detached)
            pts_clip = torch.stack([pts_clip_x, pts_clip_y, pts_clip_z], dim=-1)

        # clip space (OpenGL) to normalized coordinates (torch.nn.functional.grid_sample)
        pts_x_screen = pts_clip[..., 0]
        pts_y_screen = - pts_clip[..., 1]
        pts_z_screen = pts_clip[..., 2]   # TODO, DRTK document says this should be the actual z coords, change later

        pts_screen = torch.stack([pts_x_screen,
                                  pts_y_screen,
                                  pts_z_screen], dim=-1)
        
        return pts_screen

    @classmethod
    def proj_points_to_drtk_screen_batched(cls,
                                           proj_mat_batched: torch.Tensor,
                                           w2c_mat_batched: torch.Tensor,
                                           H: int,
                                           W: int,
                                           pts: torch.Tensor,
                                           detach_z: bool):
        """
        pts: [N, 3], unbatched points

        out: [B, N, 3], batched DRTK screen space coordinates, (-0.5, -0.5) to (W-0.5, H-0.5)
        """
        assert len(pts.shape) == 2, "input points must be unbatched, since cameras are batched"
        assert pts.shape[-1] == 3

        # full projection
        pts_wld_homog = torch.nn.functional.pad(pts, (0,1), mode='constant', value=1)       # [N, 4]
        pts_cam_homog = torch.einsum('byx,nx->bny', w2c_mat_batched, pts_wld_homog)    # [B, N, 4]
        
        if not detach_z:
            pts_clip_homog = torch.einsum('byx,bnx->bny', proj_mat_batched,  pts_cam_homog)    # [B, N, 4]
            pts_clip = pts_clip_homog[..., :3] / pts_clip_homog[..., [3]]   # [B, N, 3]
        else:
            pts_cam_x, pts_cam_y, pts_cam_z, _ = pts_cam_homog.unbind(dim=-1)   # [B, N]
            pts_cam_z_detached = pts_cam_z.detach()
            pts_clip_x = -(proj_mat_batched[:,0,0,None] * pts_cam_x / pts_cam_z_detached + proj_mat_batched[:,0,2,None])  # [B, N]
            pts_clip_y = -(proj_mat_batched[:,1,1,None] * pts_cam_y / pts_cam_z_detached + proj_mat_batched[:,1,2,None])
            pts_clip_z = -(proj_mat_batched[:,2,2,None] + proj_mat_batched[:,2,3,None] / pts_cam_z_detached)
            pts_clip = torch.stack([pts_clip_x, pts_clip_y, pts_clip_z], dim=-1)    # [B, N, 3]

        # clip space (OpenGL) to screen space (DRTK)
        pts_x_screen = (pts_clip[..., 0] + 1) / 2 * W - 0.5
        pts_y_screen = (1 - pts_clip[..., 1]) / 2 * H - 0.5
        pts_z_screen = pts_clip[..., 2]   # TODO, DRTK document says this should be the actual z coords, change later

        pts_screen = torch.stack([pts_x_screen,
                                  pts_y_screen,
                                  pts_z_screen], dim=-1)    # [B, N, 3]
        
        return pts_screen

class BatchDRTKCamera(torch.nn.Module):
    def __init__(self, cameras: List[DRTKCamera]):
        """
        proj and w2c follows OpenGL convention
        """
        super().__init__()

        self.H = cameras[0].H
        self.W = cameras[0].W

        proj_mats = []
        w2c_mats = []

        for cam in cameras:
            assert cam.H == self.H
            assert cam.W == self.W

            proj_mats.append(cam.proj_mat.clone())
            w2c_mats.append(cam.w2c_mat.clone())

        self.proj_mat = torch.nn.Buffer(torch.stack(proj_mats, dim=0))  # [B, 4, 4]
        self.w2c_mat = torch.nn.Buffer(torch.stack(w2c_mats, dim=0))  # [B, 4, 4]

    @property
    def batch_size(self) -> int:
        return self.proj_mat.shape[0]

    def proj_points_to_drtk_screen(self, pts: torch.Tensor, detach_z: bool):
        """
        pts: [B, N, 3] or [N, 3]

        out: [B, N, 3], batched DRTK screen space coordinates, (-0.5, -0.5) to (W-0.5, H-0.5)
        """
        assert pts.shape[-1] == 3
        assert (len(pts.shape) == 2) or (len(pts.shape) == 3 and pts.shape[0] == self.batch_size), \
            "input points must be unbatched, or have the same batch size the batched cameras"
        
        if len(pts.shape) == 2:
            pts = pts[None].expand(self.batch_size, -1, -1) # [B, N, 4]

        # full projection
        pts_wld_homog = torch.nn.functional.pad(pts, (0,1), mode='constant', value=1) # [B, N, 4]
        pts_cam_homog = torch.einsum('byx,bnx->bny', self.w2c_mat, pts_wld_homog)     # [B, N, 4]
        
        if not detach_z:
            pts_clip_homog = torch.einsum('byx,bnx->bny', self.proj_mat,  pts_cam_homog)    # [B, N, 4]
            pts_clip = pts_clip_homog[..., :3] / pts_clip_homog[..., [3]]   # [B, N, 3]
        else:
            pts_cam_x, pts_cam_y, pts_cam_z, _ = pts_cam_homog.unbind(dim=-1)   # [B, N]
            pts_cam_z_detached = pts_cam_z.detach()
            pts_clip_x = -(self.proj_mat[:,0,0,None] * pts_cam_x / pts_cam_z_detached + self.proj_mat[:,0,2,None])  # [B, N]
            pts_clip_y = -(self.proj_mat[:,1,1,None] * pts_cam_y / pts_cam_z_detached + self.proj_mat[:,1,2,None])
            pts_clip_z = -(self.proj_mat[:,2,2,None] + self.proj_mat[:,2,3,None] / pts_cam_z_detached)
            pts_clip = torch.stack([pts_clip_x, pts_clip_y, pts_clip_z], dim=-1)    # [B, N, 3]

        # clip space (OpenGL) to screen space (DRTK)
        pts_x_screen = (pts_clip[..., 0] + 1) / 2 * self.W - 0.5
        pts_y_screen = (1 - pts_clip[..., 1]) / 2 * self.H - 0.5
        pts_z_screen = pts_clip[..., 2]   # TODO, DRTK document says this should be the actual z coords, change later

        pts_screen = torch.stack([pts_x_screen,
                                  pts_y_screen,
                                  pts_z_screen], dim=-1)    # [B, N, 3]
        
        return pts_screen
    
class GSCamera:
    """
    camera frame uses the OpenGL convention

    the z clip coord is slightly different from OpenGL 
    """
    def __init__(self,
                 tanfov_x: float,
                 tanfov_y: float,
                 world_view_transform: torch.Tensor,
                 projection_matrix: torch.Tensor,
                 full_proj_transform: torch.Tensor,
                 camera_center: torch.Tensor,
                 H: int, W: int):

        self.tanfov_x = tanfov_x
        self.tanfov_y = tanfov_y
        self.world_view_transform = world_view_transform
        self.projection_matrix = projection_matrix
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        self.H = H
        self.W = W

    @classmethod
    def from_intr_extr_torch(cls, intr_3x3: torch.Tensor, extr_4x4: torch.Tensor, H: int, W: int):
        
        intr = intr_3x3
        extr = extr_4x4

        # Set up rasterization configuration
        fov_x = focal2fov(intr[0, 0].item(), W)
        fov_y = focal2fov(intr[1, 1].item(), H)
        tanfov_x = math.tan(fov_x * 0.5)
        tanfov_y = math.tan(fov_y * 0.5)

        world_view_transform = extr.T
        projection_matrix = get_projection_matrix_3dgs(znear = 0.1, zfar = 100, fov_x = fov_x, fov_y = fov_y, K = intr, img_w = W, img_h = H, array_package=torch, device=intr_3x3.device).T
        full_proj_transform = world_view_transform @ projection_matrix
        camera_center = torch.linalg.inv(extr)[:3, 3]
        return cls(tanfov_x, tanfov_y, world_view_transform, projection_matrix, full_proj_transform, camera_center, H, W)


class NvdiffrecmcCamera(torch.nn.Module):
    def __init__(self, mvp, campos, H, W):
        """
        full_proj_matrix: composite of projection and world2view
        """
        super().__init__()
        self.mvp = torch.nn.Buffer(torch.from_numpy(mvp))  # [4, 4]
        self.campos = torch.nn.Buffer(torch.from_numpy(campos))  # [3,]
        self.H = H
        self.W = W

class UnifiedCamera:
    def __init__(self, K: np.ndarray, R: np.ndarray, T: np.ndarray, H: int, W: int, name: str=None):
        """
        Note: K, R, T use the OpenCV conventions, i.e., in the camera frame
          +x <=> right
          +y <=> down
          +z <=> front
        When exporting these to the opengl format, you need to change K, R, T for all
        """
        self.K: np.ndarray = K.copy()  # screen space (not NDC space)
        self.R: np.ndarray = R.copy()  # w2c
        self.T: np.ndarray = T.copy()  # w2c
        self.H: int = H
        self.W: int = W
        self.name: str = name

        w2c = np.eye(4)
        w2c[:3, :3] = self.R
        w2c[:3, 3] = self.T
        self.c2w = np.linalg.inv(w2c)

    def scale(self, s_x, s_y):
        K = self.K.copy()
        K[0] = self.K[0] * s_x
        K[1] = self.K[1] * s_y
        H = int(self.H * s_x)
        W = int(self.W * s_y)
        return UnifiedCamera(K, self.R.copy(), self.T.copy(), H, W)
    
    def crop(self, x_crop_low: int, y_crop_low: int, new_H: int, new_W: int):
        K = self.K.copy()
        K[0, 2] = self.K[0, 2] - x_crop_low
        K[1, 2] = self.K[1, 2] - y_crop_low
        H = new_H
        W = new_W
        return UnifiedCamera(K, self.R.copy(), self.T.copy(), H, W)

    @property
    def front(self):
        return self.c2w[:3, 2]

    @property
    def center(self):
        return self.c2w[:3, 3]

    def proj_world2camera_opencv(self, pts: np.ndarray):
        """
        pts: [B, N, 3]
        """
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3

        # full projection
        return np.einsum('ij,nj->ni', self.R, pts) + self.T

    @classmethod
    def from_lookat(cls, center, lookat, up, fov_x, fov_y, fov_mode, K, H, W, name=None) -> Self:
        """
        Provide either (fov_x, fov_y) or K

        OpenCV convention w2c matrix
        priority: front > up > right
        """
        
        
        # camera pose
        front = lookat - center
        front /= np.linalg.norm(front).clip(min=1e-8)
        up = up - (up @ front) * front
        down = - up / np.linalg.norm(up).clip(min=1e-8)
        right = np.linalg.cross(down, front)

        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = front
        c2w[:3, 3] = center

        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        # intrinsics
        if K is None:
            if fov_x is None or fov_y is None:
                raise NotImplementedError("Proivde either (fov_x, fov_y) or K")
            else:
                assert fov_mode in ['deg', 'rad'], f'param fov_mode must be either deg or rad, now it is {fov_mode}'

                if fov_mode == 'deg':
                    fov_x = fov_x / 180 * math.pi
                    fov_y = fov_y / 180 * math.pi
                else:
                    pass
                
                f_x = W / (2 * math.tan(fov_x / 2))
                f_y = H / (2 * math.tan(fov_y / 2))
                c_x = W / 2
                c_y = H / 2

                K = np.eye(3)
                K[0,0] = f_x
                K[1,1] = f_y
                K[0,2] = c_x
                K[1,2] = c_y
        else:
            pass    # leaving K as is

        return cls(K, R, T, H, W, name)

    @classmethod
    def from_intr_extr(cls, intr_mat_3x3, extr_mat_3x4, h, w, name=None) -> Self:
        """
        both intr 3x3 and extr 3x4 (w2c) follow the opencv convention
        """
        R = extr_mat_3x4[:, :3]
        T = extr_mat_3x4[:, 3]
        K = intr_mat_3x3
        return cls(K, R, T, h, w, name)

    @classmethod
    def from_4d_dress(cls, intr_mat_3x3, extr_mat_3x4, h, w, name=None) -> Self:
        return cls.from_intr_extr(intr_mat_3x3, extr_mat_3x4, h, w)
        # R = extr_mat_3x4[:, :3]
        # T = extr_mat_3x4[:, 3]
        # K = intr_mat_3x3
        # return cls(K, R, T, h, w, name)
    
    @classmethod
    def from_avatarrex(cls, cam_dict, name=None) -> Self:
        R = np.array(cam_dict['R']).reshape(3, 3)
        T = np.array(cam_dict['T']).reshape(-1)
        K = np.array(cam_dict['K']).reshape(3, 3)
        w, h = cam_dict['imgSize']
        return cls(K, R, T, h, w, name)

    @classmethod
    def from_actorshq(cls, w, h, rx, ry, rz, tx, ty, tz, fx, fy, px, py, name=None) -> Self:
        assert_ints(w, h)
        assert_floats(rx, ry, rz, tx, ty, tz, fx, fy, px, py)

        extr_mat = np.identity(4, np.float32)
        extr_mat[:3, :3] = cv2.Rodrigues(np.array([rx, ry, rz], np.float32))[0]
        extr_mat[:3, 3] = np.array([tx, ty, tz], np.float32)
        extr_mat = np.linalg.inv(extr_mat)

        intr_mat = np.identity(3, np.float32)
        intr_mat[0, 0] = fx * w
        intr_mat[0, 2] = px * w
        intr_mat[1, 1] = fy * h
        intr_mat[1, 2] = py * h

        return cls(intr_mat, extr_mat[:3, :3], extr_mat[:3, 3], h, w, name)

    def to_idr_format(self):
        world_mat_3x4 = self.K @ np.concatenate([self.R, self.T.reshape(3, 1)], axis=-1)
        world_mat_4x4 = np.eye(4)
        world_mat_4x4[:3, :4] = world_mat_3x4
        return world_mat_4x4
    
    def to_drtk_format(self) -> DRTKCamera:
        # OpenCV extrinsics to OpenGL world2view
        w2c_opencv = np.eye(4)
        w2c_opencv[:3, :3] = self.R
        w2c_opencv[:3, 3] = self.T

        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_opengl = c2w_opencv.copy()
        c2w_opengl[:, 1] *= -1
        c2w_opengl[:, 2] *= -1
        w2c_opengl = np.linalg.inv(c2w_opengl)
        
        # OpenCV intrinsics to OpenGL projection
        proj_opengl = opencv_intrinsics_to_opengl_projection(znear = 0.1, zfar = 100, K = self.K, img_w = self.W, img_h = self.H)
        
        return DRTKCamera(proj_opengl, w2c_opengl, self.H, self.W)
    
    def to_nvdiffrecmc_format(self) -> NvdiffrecmcCamera:
        # OpenCV extrinsics to OpenGL world2view
        w2c_opencv = np.eye(4)
        w2c_opencv[:3, :3] = self.R
        w2c_opencv[:3, 3] = self.T

        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_opengl = c2w_opencv.copy()
        c2w_opengl[:, 1] *= -1
        c2w_opengl[:, 2] *= -1
        w2c_opengl = np.linalg.inv(c2w_opengl)
        
        # OpenCV intrinsics to OpenGL projection
        proj_opengl = opencv_intrinsics_to_opengl_projection(znear = 0.1, zfar = 100, K = self.K, img_w = self.W, img_h = self.H)
        proj_opengl[1] *= -1    # flip y

        mvp = proj_opengl @ w2c_opengl
        campos = self.center
        
        return NvdiffrecmcCamera(mvp, campos, self.H, self.W)

    def to_3dgs_format(self, znear=.1, zfar=100.) -> GSCamera:

        
        extr = np.identity(4, np.float32)
        extr[:3, :3] = self.R
        extr[:3, 3] = self.T

        # Set up rasterization configuration
        fov_x = focal2fov(self.K[0, 0].item(), self.W)
        fov_y = focal2fov(self.K[1, 1].item(), self.H)
        tanfov_x = math.tan(fov_x * 0.5)
        tanfov_y = math.tan(fov_y * 0.5)

        world_view_transform = extr.T
        projection_matrix = get_projection_matrix_3dgs(znear = znear, zfar = zfar, fov_x = fov_x, fov_y = fov_y, K = self.K, img_w = self.W, img_h = self.H).T
        full_proj_transform = world_view_transform @ projection_matrix
        camera_center = np.linalg.inv(extr)[:3, 3]

        return GSCamera(tanfov_x, tanfov_y,
                        torch.from_numpy(world_view_transform),
                        torch.from_numpy(projection_matrix),
                        torch.from_numpy(full_proj_transform),
                        torch.from_numpy(camera_center),
                        self.H, self.W)