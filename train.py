import os,time
import jittor as jt
import argparse
from random import randint
import random
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer.gauss_render import GaussianRenderer
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from tensorboardX import SummaryWriter



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset) # 用于确定输出的位置
    gaussians = GaussianModel(dataset.sh_degree) # 创建gaussian对象
    scene = Scene(dataset, gaussians) # 创建scene对象
    gaussians.training_setup(opt) # 设置gaussian对象的训练参数
    if checkpoint:
        model_params, first_iter = jt.load(checkpoint)['gaussian_param'],jt.load(checkpoint)['iter']
        for key in model_params.keys():
            if key != "optimizer_state" and key != "active_sh_degree" and key != "spatial_lr_scale":
                model_params[key] = jt.array(model_params[key])
            elif key == "optimizer_state":
                for i,key2 in enumerate(model_params[key]['defaults']['param_groups']):
                    model_params[key]['defaults']['param_groups'][i]['params'][0] = jt.array(model_params[key]['defaults']['param_groups'][i]['params'][0])
                    model_params[key]['defaults']['param_groups'][i]['values'][0] = jt.array(model_params[key]['defaults']['param_groups'][i]['values'][0])
                    model_params[key]['defaults']['param_groups'][i]['m'][0] = jt.array(model_params[key]['defaults']['param_groups'][i]['m'][0])
                    model_params[key]['defaults']['param_groups'][i]['grads'][0] = jt.array(model_params[key]['defaults']['param_groups'][i]['grads'][0])
        gaussians.restore(model_params, opt)
    
    viewpoint_stack = None # 用于存储视角信息
    ema_loss_for_log = 0.0 # 用于计算每个iteration的loss
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") # 用于显示进度条
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   # 测试与网络gui交互      
        
        iter_start = time.time()

        gaussians.update_learning_rate(iteration)

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
        gaussian_renderer = GaussianRenderer(active_sh_degree=gaussians.active_sh_degree,white_bkgd=dataset.white_background) # 创建gaussian_renderer对象
        render_pkg = gaussian_renderer.forward(viewpoint_cam,gaussians) # 调用gaussian_renderer中的render函数进行光栅化渲染，返回的是tensor字典，需要转换为jittor，在下面操作时转换
        image,viewspace_point_tensor, visibility_filter, radii = render_pkg["render"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"] # 获取渲染结果
        gaussians.screenspace_points.assign(viewspace_point_tensor) # 更新视空间坐标
       
        # Loss
        gt_image = viewpoint_cam.original_image.astype(jt.float32) # 获取原始图像
        Ll1 = l1_loss(image, gt_image) # 计算loss L1
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # 计算loss SSIM
        gaussians.optimizer.backward(loss) # 反向传播
        grad = gaussians.screenspace_points.opt_grad(gaussians.optimizer) # 获取梯度
        try:
            # 你的代码
            viewspace_point_tensor_grad = jt.concat([grad, jt.zeros((grad.shape[0], 1), dtype=grad.dtype)], dim=1)
        except RuntimeError:
            print("Error occurred, the shape of grad is: ", grad.shape)
            raise
        iter_end=time.time() # 用于计算每个iteration的时间
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

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad() # 梯度清零

        with jt.no_grad(): 
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer,iteration,Ll1,loss, l1_loss, iter_end-iter_start,testing_iterations, scene, gaussian_renderer.forward)    
            
            if (iteration % saving_iterations == 0):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration % checkpoint_iterations == 0):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                jt.save({"gaussian_param":gaussians.capture(),"iter":iteration}, scene.model_path + "/chkpnt" + str(iteration) + ".pkl")

        # test to decrease the cuda memory
        # del image,viewspace_point_tensor,gt_image,grad, render_pkg,loss,viewspace_point_tensor_grad,Ll1,visibility_filter, radii # 删除不需要的变量
        # jt.clean_graph()
        # jt.sync_all()
        jt.gc() # 清理图，同步所有设备，进行垃圾回收
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
    
    # Set up tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer
    

    

def training_report(tb_writer,iteration,Ll1,loss,l1_loss,elapsed, testing_iterations, scene : Scene, renderFunc):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    # Report test and samples of training set
    if iteration % testing_iterations == 0:
        jt.gc() # 清理图，同步所有设备，进行垃圾回收
        # jt.display_memory_info()
        
        # 记录梯度范数
        grads = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
        grads[grads.isnan()] = 0.0 # 计算梯度并将梯度张量中的NaN值替换为0
        grads_record = jt.norm(grads, dim=-1).mean().numpy()
        tb_writer.add_scalar('grads_norm', grads_record, iteration)
         # 记录当前xyz的学习率
        current_lr = scene.gaussians.optimizer.param_groups[0]['lr']
        tb_writer.add_scalar('Learning Rate', current_lr, iteration)


        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx,viewpoint in enumerate(config['cameras']):
                    image = jt.clamp(renderFunc(viewpoint, scene.gaussians)["render"], min_v=0.0, max_v=1.0)
                    gt_image = jt.clamp(viewpoint.original_image.astype(jt.float32), min_v=0.0, max_v=1.0)
                    if tb_writer and (idx<5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None].numpy(), global_step=iteration)
                        if iteration == testing_iterations:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None].numpy(), global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test.numpy(), iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test.numpy(), iteration)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity.numpy(), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)        
        jt.gc() # 清理图，同步所有设备，进行垃圾回收
        # jt.display_memory_info()

if __name__ == "__main__":
    
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution=0
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", type=int, default=500)
    parser.add_argument("--save_iterations", type=int, default=500)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", type=int, default=500)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # # use to Multi-card parallel training
    # parser.add_argument("--local_rank", type=int,default=-1)


    args = parser.parse_args(sys.argv[1:])

    # torch.cuda.set_device(args.local_rank)  # before your code runs
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
