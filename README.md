# JGaussian
Use Jittor to recurrent Gaussain Splatting

Project is in the process...

## how to use?

This project is mainly based on official implemention of Gaussian Spaltiing , the args are all similar to official implemention.To get detailed explanation of args , you can click [it](https://github.com/graphdeco-inria/gaussian-splatting) to look detailed imformation

After your training , if you want it to look in web , you can download your output folders' .ply file and drag it to [there](https://antimatter15.com/splat/) to look it in the 3D scene .

Noticed that because of CUDA memory , we all reduce the image resolution to 0.125 by using args of "-r" "8"

Now we support 1000 iterations in a TiTanXP in 1 hours around , and its SSIM is 0.47, PSNR is 19.71

compare to purely pytorch Gaussian-splatting [Gaussian_torch](https://github.com/xiazhi1/Gaussian_torch)

1000 iterations in a TiTanXP in 1.5 hours around , and its SSIM is 0.45, PSNR is 19.68

JGaussian is faster and better which shows jittor's strength in speed

compare to official Gaussian-splatting [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

7000 iterations in a TiTanXP in 9 minutes around , and its SSIM is 0.81, PSNR is 26

CUDA rasterizer is more faster and better than pytorch or jittor rasterizer.

## Reference

[https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

[https://github.com/hbb1/torch-splatting](https://github.com/hbb1/torch-splatting)

[https://github.com/Jittor/jittor](https://github.com/Jittor/jittor)

[https://github.com/antimatter15/splat](https://github.com/antimatter15/splat)
