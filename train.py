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

import os,time
import jittor as jt
import argparse
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import  network_gui
from gaussian_renderer.gauss_render import GaussianRenderer
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# tensorboard 先不管
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset) # 用于确定输出的位置和在tensorboard上记录的参数
    gaussians = GaussianModel(dataset.sh_degree) # 创建gaussian对象
    scene = Scene(dataset, gaussians) # 创建scene对象
    gaussians.training_setup(opt) # 设置gaussian对象的训练参数
    if checkpoint:
        (model_params, first_iter) = jt.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    if not dataset.white_background:
        gaussian_renderer = GaussianRenderer(white_bkgd=False)
    gaussian_renderer = GaussianRenderer()
    background = jt.array(bg_color, dtype=jt.float32) # 背景颜色

    viewpoint_stack = None # 用于存储视角信息
    ema_loss_for_log = 0.0 # 用于计算每个iteration的loss
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") # 用于显示进度条
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   # 测试与网络gui交互      
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = gaussian_renderer(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((jt.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None 
        iter_start = time.time()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = gaussian_renderer.forward(viewpoint_cam,gaussians) # 调用gaussian_renderer中的render函数进行光栅化渲染，返回的是tensor字典，需要转换为jittor，在下面操作时转换
        image,viewspace_point_tensor, visibility_filter, radii = render_pkg["render"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"] # 获取渲染结果


        # render_pkg = render(viewpoint_cam, gaussians, pipe, background) # 调用gaussian_renderer中的render函数进行光栅化渲染，返回的是tensor字典，需要转换为jittor，在下面操作时转换
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # 如果想把image转为jt.var必须先转为numpy，但是带梯度的tensor转为numpy会被丢弃梯度，进而导致无法反向传播
        # 最后得出的结论是因为jittor没有C++ API 无法与cuda交互进行渲染，导致项目无法进行下去，因为无梯度的tensor无法进行反向传播

        gaussians.screenspace_points.assign(viewspace_point_tensor) # 更新视空间坐标
        # gaussians.optimizer.zero_grad() # 梯度清零

        # # # test code to verify where grad is lost
        # loss = image.sum()
        # gaussians.optimizer.backward(image) # 反向传播 发现此处梯度也是很多0.....

        # Loss
        gt_image = viewpoint_cam.original_image.astype(jt.float32) # 获取原始图像
        Ll1 = l1_loss(image, gt_image) # 计算loss L1
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # 计算loss SSIM
        gaussians.optimizer.backward(loss) # 反向传播
        # grad = jt.grad(loss, viewspace_point_tensor)
        grad = gaussians.screenspace_points.opt_grad(gaussians.optimizer) # 获取梯度
         
        # test to get the non zero grad
        # non_zero_row_indices = jt.any(grad != 0, dim=1)
        # non_zero_rows = grad[non_zero_row_indices]
        # print(non_zero_rows)


        viewspace_point_tensor_grad = jt.concat([grad, jt.zeros((grad.shape[0], 1), dtype=grad.dtype)], dim=1) # 由于视空间坐标是三维的，而梯度是二维的，所以需要在梯度后面加一个0
        viewspace_point_tensor_grad = jt.array(viewspace_point_tensor_grad, dtype=viewspace_point_tensor.dtype) # 转换为tensor
        iter_end=time.time() # 用于计算每个iteration的时间

        
    

        with jt.no_grad(): 
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_end-iter_start, testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter: # 如果iteration小于densify_until_iter，就进行高斯点云的密度增加
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = jt.maximum(gaussians.max_radii2D[visibility_filter], radii[visibility_filter]) # 更新最大半径
                gaussians.add_densification_stats(viewspace_point_tensor_grad,visibility_filter) # 更新视空间坐标和可见性

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: # 如果当前迭代大于指定的开始密集化的迭代数，并且当前迭代是密集化间隔的倍数，那么就进行密集化和修剪操作
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter): # 如果当前迭代是透明度重置间隔的倍数，或者是白色背景并且当前迭代等于指定的开始密集化的迭代数，那么就重置透明度
                    gaussians.reset_opacity()

                

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                jt.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad() # 梯度清零


        # jt.clean_graph()
        # jt.sync_all()
        # jt.gc()
        # jt.display_memory_info()
        

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # # Create Tensorboard writer
    # tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        # torch.cuda.empty_cache() # Jittor自动释放
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = jt.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = jt.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        # torch.cuda.empty_cache()

if __name__ == "__main__":
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'
    jt.flags.use_cuda = 1
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # # use to Multi-card parallel training
    # parser.add_argument("--local_rank", type=int,default=-1)


    args = parser.parse_args(sys.argv[1:])

    # torch.cuda.set_device(args.local_rank)  # before your code runs
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
