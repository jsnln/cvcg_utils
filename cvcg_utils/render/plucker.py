import numpy as np
import scipy
from scipy.sparse.linalg import svds, lsqr, LinearOperator

def raymap2center(rayd_list: np.ndarray, raym_list: np.ndarray):
    """
    rayd_list: [N_rays, 6]
    """
    rays_d = rayd_list
    rays_m = raym_list
    
    def matvec(p: np.ndarray):
        """
        p: [3,]
        """
        p_cross_d = np.cross(p[None], rays_d, axis=-1)  # [N_rays, 3]
        return p_cross_d.reshape(-1)
    
    def rmatvec(q: np.ndarray):
        """
        q: [N_rays*3]
        """
        q = q.reshape(-1, 3)
        d_cross_q = np.cross(rays_d, q, axis=-1).sum(0)
        return d_cross_q
    
    raymap2center_op = LinearOperator((rays_d.shape[0]*3, 3), matvec=matvec, rmatvec=rmatvec)
    center = lsqr(raymap2center_op, rays_m.reshape(-1))[0]

    return center

def raymap2K(rayd_local_list: np.ndarray, uv1_list: np.ndarray):
    """
    rayd_local_list: [N_rays, 3]
    uv1_list: [N_rays, 3] (homogeneous)
    """
    rayd_local_list = rayd_local_list / rayd_local_list[:, [2]]
    
    # linop: K \mapsto K @ rays_d.T, but compressed
    def matvec(K_c: np.ndarray):
        """
        K_c: [4,], (fx, fy, cx, cy)
        """
        u_proj = rayd_local_list[:, 0] * K_c[0] + K_c[2]
        v_proj = rayd_local_list[:, 1] * K_c[1] + K_c[3]
        return np.stack([u_proj, v_proj], axis=-1).reshape(-1)  # [N_rays*2]
    
    def rmatvec(uv: np.ndarray):
        """
        uv: [N_rays*2]
        """
        uv = uv.reshape(-1, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        K_0 = rayd_local_list[:, 0] * u
        K_1 = rayd_local_list[:, 1] * v
        K_2 = u
        K_3 = v
        K = np.stack([K_0, K_1, K_2, K_3], axis=-1).sum(0)  # [N_rays, 4]
        return K  # [4]

    raymap2K_op = LinearOperator((rayd_local_list.shape[0]*2, 4), matvec=matvec, rmatvec=rmatvec)
    K = lsqr(raymap2K_op, uv1_list[:, :2].reshape(-1))[0]

    return K

def raymap2KRt(rayd_list: np.ndarray, raym_list: np.ndarray, uv1_list: np.ndarray):
    """
    rays: [N_rays, 6]
    uv1_list: [N_rays, 3] (homogeneous)
    """
    rays_d = rayd_list
    rays_m = raym_list

    cam_center = raymap2center(rays_d, rays_m)
    
    def matvec(P: np.ndarray):
        """
        R: [9,] <=> [3, 3], rotation part of the projection matrix
        """
        P = P.reshape(3, 3)
        Pd = np.einsum('ij,nj->ni', P, rays_d)
        BP = np.cross(uv1_list, Pd, axis=-1)    # [Nrays, 3]

        return BP.reshape(-1)
    
    def rmatvec(r: np.ndarray):
        """
        r: [N_rays*3]
        """
        r = r.reshape(-1, 3)    # [N, 3]
        m_uv1_cross_r = -np.cross(uv1_list, r, axis=-1)         # [N, 3]
        BTr = np.einsum('ni,nj->nij', m_uv1_cross_r, rays_d).sum(0)  # [3, 3]

        return BTr.reshape(-1)
    
    raymap2P_op = LinearOperator((rays_d.shape[0]*3, 9), matvec=matvec, rmatvec=rmatvec)
    # (u, s, vh) = svds(raymap2P_op, k=9, maxiter=100, solver='propack', return_singular_vectors=False)
    # arpack has really bad convergence, use propack instead
    (_, ss, vhs) = svds(raymap2P_op, which='SM', k=1, return_singular_vectors='vh')
    vhs = vhs.reshape(3, 3)
    
    # step 0: rectify overall orientation
    # NOTE both vhs and -vhs are valid singular vectors
    # we need the one with positive determinant because det(KR) > 0
    if np.linalg.det(vhs) < 0:
        vhs = -vhs

    K_prelim_solved, R_prelim_solved = scipy.linalg.rq(vhs)
    assert np.linalg.det(R_prelim_solved) > 0
    # NOTE QR result is not usable yet
    # K is not usable here due to scale ambiguity
    # R may have still a wrong orientation

    # NOTE step 1: further rectify orientation
    if K_prelim_solved[0,0] < 0:    # need to flip first row of R
        K_prelim_solved[:, 0] *= -1
        R_prelim_solved[0] *= -1

    if K_prelim_solved[1,1] < 0:    # need to flip second row of R
        K_prelim_solved[:, 1] *= -1
        R_prelim_solved[1] *= -1
    
    if np.linalg.det(R_prelim_solved) < 0:    # need to flip third row of R
        K_prelim_solved[:, 2] *= -1
        R_prelim_solved[2] *= -1

    # NOTE step 2: rectify scale
    K_prelim_solved /= K_prelim_solved[2,2] # 
    
    K = K_prelim_solved
    R = R_prelim_solved
    T = - R @ cam_center

    return K, R, T

def refine_KRT_with_known_focal(K, R, T, focal, H, W):
    new_K = K.copy()
    new_K[0, 1] = 0
    new_K[0, 0] = focal
    new_K[1, 1] = focal
    new_K[0, 2] = W / 2
    new_K[1, 2] = H / 2

    cam_center = - R.T @ T
    
    new_R_cand = np.linalg.inv(new_K) @ K @ R

    K_residual, new_R = scipy.linalg.rq(new_R_cand)
    assert np.linalg.det(new_R) > 0
    # NOTE QR result is not usable yet
    # K is not usable here due to scale ambiguity
    # R may have still a wrong orientation

    # NOTE further rectify orientation
    if K_residual[0,0] < 0:    # need to flip first row of R
        K_residual[:, 0] *= -1
        new_R[0] *= -1

    if K_residual[1,1] < 0:    # need to flip second row of R
        K_residual[:, 1] *= -1
        new_R[1] *= -1
    
    if np.linalg.det(new_R) < 0:    # need to flip third row of R
        K_residual[:, 2] *= -1
        new_R[2] *= -1

    new_T = - new_R @ cam_center

    return new_K, new_R, new_T




def solve_PnP(xyz_list: np.ndarray, uv1_list: np.ndarray):
    """
    xyz_list: [N_rays, 3]
    uv1_list: [N_rays, 3] (homogeneous)
    """
    xyz_list_homog = np.pad(xyz_list_homog, ((0,0), (0,1)), constant_values=1)  # [N, 4]
    
    def matvec(P: np.ndarray):
        """
        P: [12,] <=> [3, 4], full projection matrix
        """
        P = P.reshape(3, 4)
        Pxyz = np.einsum('ij,nj->ni', P[:3, :3], xyz_list) + P[:3, 3]
        BP = np.cross(uv1_list, Pxyz, axis=-1)    # [Nrays, 3]

        return BP.reshape(-1)
    
    def rmatvec(r: np.ndarray):
        """
        r: [N_rays*3]
        """
        r = r.reshape(-1, 3)    # [N, 3]
        m_uv1_cross_r = -np.cross(uv1_list, r, axis=-1)         # [N, 3]
        BTr = np.einsum('ni,nj->nij', m_uv1_cross_r, xyz_list_homog).sum(0)  # [3, 4]

        return BTr.reshape(-1)
    
    raymap2P_op = LinearOperator((xyz_list.shape[0]*3, 12), matvec=matvec, rmatvec=rmatvec)
    # (u, s, vh) = svds(raymap2P_op, k=9, maxiter=100, solver='propack', return_singular_vectors=False)
    # arpack has really bad convergence, use propack instead
    (_, ss, vhs) = svds(raymap2P_op, which='SM', k=1, return_singular_vectors='vh')
    vhs = vhs.reshape(3, 4)
    
    # step 0: rectify overall orientation
    # NOTE both vhs and -vhs are valid singular vectors
    # we need the one with positive determinant because det(KR) > 0
    if np.linalg.det(vhs[:3, :3]) < 0:
        vhs = -vhs

    K_prelim_solved, R_prelim_solved = scipy.linalg.rq(vhs[:3, :3])
    RT_prelim_solved = np.linalg.inv(K_prelim_solved) @ vhs
    assert np.linalg.det(R_prelim_solved) > 0
    # NOTE QR result is not usable yet
    # K is not usable here due to scale ambiguity
    # R may have still a wrong orientation

    # NOTE step 1: further rectify orientation
    if K_prelim_solved[0,0] < 0:    # need to flip first row of R
        K_prelim_solved[:, 0] *= -1
        RT_prelim_solved[0] *= -1

    if K_prelim_solved[1,1] < 0:    # need to flip second row of R
        K_prelim_solved[:, 1] *= -1
        RT_prelim_solved[1] *= -1
    
    if np.linalg.det(RT_prelim_solved[:3, :3]) < 0:    # need to flip third row of R
        K_prelim_solved[:, 2] *= -1
        RT_prelim_solved[2] *= -1

    # NOTE step 2: rectify scale
    K_prelim_solved /= K_prelim_solved[2,2] # 
    
    K = K_prelim_solved
    R = RT_prelim_solved[:3, :3]
    T = RT_prelim_solved[:3, 3]

    return K, R, T