from typing import Self
import cv2
import math
import numpy as np
import torch

def assert_ints(*args):
    for arg in args:
        assert isinstance(arg, int)

def assert_floats(*args):
    for arg in args:
        assert isinstance(arg, float)

def get_projection_matrix_3dgs(znear, zfar, fov_x=None, fov_y=None, K=None, img_h=None, img_w=None):
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

    P = np.zeros((4, 4))

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
        full_proj_matrix: composite of projection and world2view
        """
        super().__init__()
        self.proj_mat = torch.nn.Buffer(torch.from_numpy(proj_mat))  # [H, W]
        self.w2c_mat = torch.nn.Buffer(torch.from_numpy(w2c_mat))  # [H, W]
        self.H = H
        self.W = W

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
        pts_z_screen = pts_clip[..., 2]   # TODO, DRTK document says this should be the actual z coords, change later

        pts_screen = torch.stack([pts_x_screen,
                                  pts_y_screen,
                                  pts_z_screen], dim=-1)
        
        return pts_screen
    
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

    def proj_world2camera_opencv(self, pts: np.ndarray):
        """
        pts: [B, N, 3]
        """
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3

        # full projection
        return np.einsum('ij,nj->ni', self.R, pts) + self.T

    @classmethod
    def from_lookat(cls, center, lookat, up, fov_x, fov_y, K, H, W, name=None) -> Self:
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
    def from_4d_dress(cls, intr_mat_3x3, extr_mat_3x4, h, w, name=None) -> Self:
        R = extr_mat_3x4[:, :3]
        T = extr_mat_3x4[:, 3]
        K = intr_mat_3x3
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

    def to_3dgs_format(self):

        def focal2fov(focal, pixels):
            return 2 * math.atan(pixels / (2*focal))
    
        extr = np.identity(4, np.float32)
        extr[:3, :3] = self.R
        extr[:3, 3] = self.T

        # Set up rasterization configuration
        fov_x = focal2fov(self.K[0, 0].item(), self.W)
        fov_y = focal2fov(self.K[1, 1].item(), self.H)
        tanfov_x = math.tan(fov_x * 0.5)
        tanfov_y = math.tan(fov_y * 0.5)

        world_view_transform = extr.T
        projection_matrix = get_projection_matrix_3dgs(znear = 0.1, zfar = 100, fov_x = fov_x, fov_y = fov_y, K = self.K, img_w = self.W, img_h = self.H).T
        full_proj_transform = world_view_transform @ projection_matrix
        camera_center = np.linalg.inv(extr)[:3, 3]
        return fov_x, fov_y, tanfov_x, tanfov_y, world_view_transform, projection_matrix, full_proj_transform, camera_center
