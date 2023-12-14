#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import jittor as jt
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : jt.array, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # 问题最大的地方 jittor无法对非叶子节点保留梯度，尝试新建优化器，直接计算其与loss的梯度都失败，结果都为0，最后暂时放弃，先往后面走
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = jt.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype) + 0 # 创建一个和pc.get_xyz相同大小的全0张量，用于存储空间坐标的投影坐标，也就是模拟3DGaussian的投影
    try:
        screenspace_points.retain_grad() # 尝试对非叶子节点保留梯度
    except:
        pass


    # Set up rasterization configuration：设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5) # 计算视点相机的水平和垂直视场角的正切值

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.from_numpy(bg_color.numpy()).cuda().requires_grad_(True),
        scale_modifier=scaling_modifier,
        viewmatrix=torch.from_numpy(viewpoint_camera.world_view_transform.numpy()).cuda().requires_grad_(True),
        projmatrix=torch.from_numpy(viewpoint_camera.full_proj_transform.numpy()).cuda().requires_grad_(True),
        sh_degree=pc.active_sh_degree,
        campos=torch.from_numpy(viewpoint_camera.camera_center.numpy()).cuda().requires_grad_(True),
        prefiltered=False,
        debug=pipe.debug
    )# 创建光栅化配置

    rasterizer = GaussianRasterizer(raster_settings=raster_settings) # 创建光栅化器（暂时未细读）

    means3D = pc.get_xyz # 获取Gaussian的3D坐标
    means2D = screenspace_points # 获取Gaussian的2D投影坐标
    opacity = pc.get_opacity # 获取Gaussian的不透明度

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling # 获取Gaussian的缩放比例
        rotations = pc.get_rotation # 获取Gaussian的旋转角度

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None: # 如果提供了球谐函数，就用它们来计算颜色，否则就用光栅化器来计算颜色
        if pipe.convert_SHs_python: # 如果pipe.convert_SHs_python为真，就用Python来计算球谐函数
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) # 获取点云特征并处理得到球谐函数视图
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)) # 计算点云中心到相机中心的方向向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # 将方向向量归一化
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) # 计算球谐函数到RGB的转换
            colors_precomp = jt.maximum(sh2rgb + 0.5, 0.0)  # 加上0.5裁剪到[0, 1]范围内
        else:
            shs = pc.get_features # 直接获取点云特征作为球谐函数
    else:
        colors_precomp = override_color

    # Convert var to tensors to be used in rasterizer
    tensor_means3D = torch.from_numpy(means3D.numpy()).cuda().requires_grad_(True)
    tensor_means2D = torch.from_numpy(means2D.numpy()).cuda().requires_grad_(True)
    tensor_opacity = torch.from_numpy(opacity.numpy()).cuda().requires_grad_(True)
    tensor_shs = torch.from_numpy(shs.numpy()).cuda().requires_grad_(True) if shs is not None else None
    tensor_colors_precomp = torch.from_numpy(colors_precomp.numpy()).cuda().requires_grad_(True) if colors_precomp is not None else None
    tensor_scales = torch.from_numpy(scales.numpy()).cuda().requires_grad_(True) if scales is not None else None
    tensor_rotations = torch.from_numpy(rotations.numpy()).cuda().requires_grad_(True) if rotations is not None else None
    tensor_cov3D_precomp = torch.from_numpy(cov3D_precomp.numpy()).cuda().requires_grad_(True) if cov3D_precomp is not None else None
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = tensor_means3D,
        means2D = tensor_means2D,
        shs = tensor_shs,
        colors_precomp = tensor_colors_precomp,
        opacities = tensor_opacity,
        scales = tensor_scales,
        rotations = tensor_rotations,
        cov3D_precomp = tensor_cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
