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
import numpy as np
import torch
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from jittor import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_kNN.distanceMetrics import distanceMetrics
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel: # 定义Gaussian模型，初始化与Gaussian模型相关的各种属性，调用setup_functions方法设置各种激活和变换函数

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation): # 该方法用于从缩放旋转中构建协方差矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = jt.exp
        self.scaling_inverse_activation = jt.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = jt.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = jt.misc.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = jt.empty((0,))
        self._features_dc = jt.empty((0,))
        self._features_rest = jt.empty((0,))
        self._scaling = jt.empty((0,))
        self._rotation = jt.empty((0,))
        self._opacity = jt.empty((0,))
        self.max_radii2D = jt.empty((0,))
        self.xyz_gradient_accum = jt.empty((0,))
        self.denom = jt.empty((0,))
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return jt.concat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale # 场景的NeRF半径在创建Gauusian时作为空间低分辨率的缩放比例
        fused_point_cloud = jt.array(np.asarray(pcd.points)).float() #将输入点云数据的坐标转换为PyTorch张量，并移动到GPU上
        fused_color = RGB2SH(jt.array(np.asarray(pcd.colors)).float()) #将输入点云数据的颜色信息转换为球谐函数表示，并移动到GPU上
        features = jt.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float() #创建一个零张量，用于存储特征信息，其形状为 (点的数量, 3, 球谐函数的维度)
        features[:, :3, 0 ] = fused_color #将点云颜色信息存储到特征张量的第一个通道中
        features[:, 3:, 1:] = 0.0 # 将特征张量的其他通道设置为零
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # 假设 pcd_points 是一个 N x 3 的张量，表示 N 个点的坐标
        pcd_points = jt.array(np.asarray(pcd.points)).float32()
        pcd_points = pcd_points[None, ...] # 转换为1 N 3的张量
        # 找到每个点的最近的 K 个点 原cuda代码里也是3
        K = 3
        distances_squared, indices = jt.knn(pcd_points, pcd_points, K)
        # 计算平均距离的平方
        average_distances_squared = distances_squared.mean(dim=-1).squeeze(dim=0)
        # 进行最小值截断，防止除以零
        dist2 = jt.maximum(average_distances_squared, 0.0000001) 
        # dist2 = jt.maximum(jt.array(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()).cpu().numpy()), 0.0000001) # 计算点云中点之间的距离平方，并进行最小值截断，防止除以零,次模块要求tensor
        scales = jt.log(jt.sqrt(dist2))[...,None].repeat(1, 3) # 计算每个点的缩放因子，以对应于点到点之间的距离 只是一个初始值，偏差不会对结果造成很大影响
        rots = jt.zeros((fused_point_cloud.shape[0], 4)) # 创建一个零张量，用于存储旋转信息，其形状为 (点的数量, 4)
        rots[:, 0] = 1 # 将旋转张量的第一个通道设置为1，其余通道设置为零

        opacities = inverse_sigmoid(0.1 * jt.ones((fused_point_cloud.shape[0], 1), dtype=jt.float)) # 创建一个张量，用于存储点的不透明度信息，其形状为 (点的数量, 1)，并将其初始化为0.1

        self._xyz = nn.Parameter(fused_point_cloud) # 将点云坐标张量转换为可优化的参数
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous()) # 将特征张量的第一个通道转换为可优化的参数(即前面提到的点云颜色特征)
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous()) # 将特征张量的其他通道转换为可优化的参数，处理方式与上面类似，都是首先选择通道，然后转置，最后用方法确保内存连续
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities) # 以上三行代码将缩放、旋转和不透明度信息转换为可优化的参数

    def training_setup(self, training_args): # 该方法用于设置训练参数和优化器,因为jittor的特殊性，这里的参数设置与PyTorch有所不同。可能会多很多的参数设置
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = jt.zeros((self.get_xyz.shape[0], 1))
        self.denom = jt.zeros((self.get_xyz.shape[0], 1)) # 创建两个零张量，用于存储点云中每个点的梯度累积和梯度累积次数，其形状都为 (点的数量, 1)
        self.screenspace_points = jt.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype) + 0 # 创建一个和pc.get_xyz相同大小的全0张量，用于存储空间坐标的投影坐标，也就是模拟3DGaussian的投影
        l = [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
            self.screenspace_points
        ]
        # 因为 jitor的优化器不支持参数组param_groups属性，不能直接为每个参数设置不同的学习率。所以暂时全部设为0.001
        self.optimizer = jt.optim.Adam(l, lr=0.1, eps=1e-15) # 创建一个Adam优化器，用于优化上面的参数列表
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps) # 创建一个学习率调度器，用于调整点云坐标的学习率

    def update_learning_rate(self, iteration): # 该方法用于更新优化器中的学习率
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(jt.min(self.get_opacity, jt.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(jt.array(xyz, dtype=jt.float))
        self._features_dc = nn.Parameter(jt.array(features_dc, jt=jt.float).transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(jt.array(features_extra, dtype=jt.float).transpose(1, 2).contiguous())
        self._opacity = nn.Parameter(jt.array(opacities, dtype=jt.float))
        self._scaling = nn.Parameter(jt.array(scales, dtype=jt.float))
        self._rotation = nn.Parameter(jt.array(rots, dtype=jt.float))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = jt.zeros_like(tensor)
                stored_state["exp_avg_sq"] = jt.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask]))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask])
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask): # 该方法用于修剪掉不需要的点
        valid_points_mask = ~mask # 生成修剪掩码
        optimizable_tensors = self._prune_optimizer(valid_points_mask) # 从优化器参数中删除不需要的点

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] # 将返回的优化器参数添加到高斯模型中

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask] # 将梯度累积和梯度累积次数，以及最大2D半径中不需要的点删除

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = jt.concat((stored_state["exp_avg"], jt.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = jt.concat((stored_state["exp_avg_sq"], jt.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(jt.concat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(jt.concat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation): #  用于在密集化处理后更新高斯模型的参数。
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d) # 将新的点云信息添加到优化器参数中
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] # 将新的优化器信息添加到高斯模型中

        self.xyz_gradient_accum = jt.zeros((self.get_xyz.shape[0], 1)).cuda()
        self.denom = jt.zeros((self.get_xyz.shape[0], 1)).cuda()
        self.max_radii2D = jt.zeros((self.get_xyz.shape[0])).cuda() # 重置梯度累积和梯度累积次数，以及最大2D半径便于后续密集化和修剪

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = jt.zeros((n_init_points)).cuda()
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = jt.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = jt.logical_and(selected_pts_mask,
                                              jt.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) # 生成掩码

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =jt.zeros((stds.size(0), 3)).cuda()
        samples = jt.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = nn.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1) 
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1) # 利用掩码提取所有满足条件的点，并进行N次切割得到新的位置、缩放和旋转信息
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = jt.concat((selected_pts_mask, jt.zeros(N * selected_pts_mask.sum(), dtype=bool).cuda()))
        self.prune_points(prune_filter) # 生成修建掩码后修剪掩码中的点

    def densify_and_clone(self, grads, grad_threshold, scene_extent): # 该方法用于对梯度张量中的点直接复制满足条件的点进行密集化
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent) #通过计算梯度范数并与阈值比较，生成掩码，再加以操作得到最终掩码
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask] #利用掩码从原始张量中提取满足条件的点

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation) #将提取的点添加到原始张量中

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size): # 该方法用于对高斯模型进行密集化和修剪。
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0 # 计算梯度并将梯度张量中的NaN值替换为0

        self.densify_and_clone(grads, max_grad, extent) # 对梯度张量中的点直接复制满足条件的点进行密集化
        self.densify_and_split(grads, max_grad, extent) # 对梯度张量中的点在满足条件的点的位置生成新的点进行密集化

        prune_mask = (self.get_opacity < min_opacity).squeeze() # 修剪掩码，用于标记不透明度小于阈值的点
        if max_screen_size: # 如果场景最大尺寸不为空，则将大点的掩码和缩放因子大于场景范围的点添加到修剪掩码中
            big_points_vs = self.max_radii2D > max_screen_size 
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask) # 修剪掩码中的点

        torch.cuda.empty_cache()   # 清空GPU缓存

    def add_densification_stats(self,viewspace_point_tensor,update_filter):
        self.xyz_gradient_accum[update_filter] += jt.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1