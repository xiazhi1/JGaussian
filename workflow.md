# workflow

## introduction

这个文档的设计是为了让作者在进行项目修改时，有一个连贯完整的调试思路和编写思路

## train.py

目前train.py主要是直接照搬GaussianSPlatting的代码，具体修改思路是暂时不管tensorboard和NetworkGui部分，先修改核心的GaussianModel与SceneModel

### bug of var

在训练过程中我们利用GPU训练，但选定后无法修改为cpu，因为GPU挂载后所有产生的var都会带上梯度，不再是原来cpu下的var，与torch中的tensor不同，这里采用的方法是先用numpy创建矩阵，再用jittor.array转为var

又不行了 还是挂不了 改不成var，因为这次返回回来的就是jittor.jittor_core.var这个没有data,也没有numpy方法，完全不行。问题在camera_utils.py的41行


**其实没有错 是我安装错jittor了，jittor最好按照1.3.6.5配 暂时没出现毛病,最新版本貌似有bug？**

### retain_grad


#### 添加到优化器求梯度

jittor和torch生成的变量var与tensor相比，有个比较重要的区别就是jittor.var没有grad属性,grad只能通过创建优化器，然后对优化器参数求jt.opt_grad，所以这里决定在train.py中专门创建一个额外的optimizer来优化在训练过程中随迭代更新但之前未被添加到GaussianModel里的参数

始终没有用，问题在于jittor无法实现对非叶子节点的梯度保留，因为jittor追求高速度，尝试了新建优化器，计算与loss的梯度，都失败，算出来的梯度都是0，最后暂时先跳过，先用tensor保留梯度,jittor的adam优化器无法对tensor计算梯度，所以最后只好都换成torch的优化器和优化参数，暂时先跑着，

也就是说这里为了求到**viewspace_point_tensor的梯度:Bug 始终求出来为0，但是源代码求出来的不是0，所以可能更新高斯点坐标，densfication这部分有问题**，因为是他是由非叶子节点screenspace_points利用retain_grad保留得到的梯度，在jittor中反复尝试均未找到好的方法，所以暂时用torch.grad保留梯度，但是其在jittor的adam优化器中无法计算梯度，所以最后用的是torch的adam优化器，但是这样好像还是求不出梯度。。。

新的解决思路，把screenspace_points放到训练参数的配置中去修改,还是不行 梯度还是0 这个地方会严重影响densification部分

[参考这里尝试改掉densification部分看能不能行的通](https://github.com/WangFeng18/3d-gaussian-splatting/blob/main/splatter.py)，感觉不太行 densification的关键就是梯度，梯度不对，怎么改都不可行

#### 尝试计算中间梯度

chatGPT:

PyTorch中的梯度计算思路，即使一个Tensor在计算一个标量的过程中并没有直接被用到，只要它与这个标量存在依赖关系，就会计算并保存它的梯度，这是因为PyTorch使用了动态计算图，并且会自动追踪和记录所有的计算操作，因此能够自动计算梯度。然而，Jittor的工作方式可能与PyTorch有所不同，因此可能无法直接实现这种梯度计算思路。可能是由于jittor的内核的计算图与torch的不太相同，所以无法自动计算梯度

无法用grad的原因：grad只能计算和loss有直接关系的变量的梯度，而viewspace_point_tensor确实没有直接和loss相关，所以计算出来都会是0，如果尝试用hook，hook函数当梯度为0时无法调用，所以也不能存储viewspace_point_tensor的梯度，也就是说jittor中或许无法实现对与loss的输入，进行其他变换后的其他变量的梯度

网上资源：

从网上搜寻资源发现：

"torch性能虽然相比上面两者明显有差距，但毕竟发展这么久的易用性还是不错的。另外笔者也可以补充一些torch的特点，比如torch的autograd库可以比较方便地拿到某个中间变量甚至输入的梯度（通常深度学习是把输入当变量求模型参数的梯度进行优化），上面两者要实现这个功能应该需要手写新算子"

[参考博客](https://zhuanlan.zhihu.com/p/635455855)

所以暂时采用torch保留梯度计算,同时把优化器改成torch的Adam,但是这样改的话，整个参与优化的loss计算的所有变量全部都要变成tensor，这样整个项目很多地方都要改成tensor,问题很大

还是想办法放到优化器参数中去算梯度试试看，排除了创建中间变量算梯度的思路

#### 优化器参数算梯度再议

如果你需要将一个需要梯度的Tensor转换为一个NumPy数组，你可以先使用.detach()方法将它从计算图中剥离，然后再调用.numpy()方法。但是，需要注意的是，这样做之后，你就不能在这个Tensor上进行反向传播了。所以从cuda源码中渲染回来的是tensor且需要梯度的类型，如果转为jittor类型需要用detach转为numpy，detach会把其从计算图上分离，之后就不能反向传播梯度了，所以这要求在反向传播梯度前都是tensor类型，这是为了与cuda渲染出来的结果一致，所以或许只能大改，把Adam优化器的变量和loss的计算全部保留torch部分。这样整个项目都很难用jittor编写，问题主要出在与光栅化器的接口部分是用的tensor，**尝试把光栅化器的接口部分改为jittor接口**

chatGPT回答：

Jittor的底层实现确实使用了C++和CUDA，但是它主要是为Python环境设计的，其主要API都是Python API。Jittor并没有提供类似于PyTorch的C++ API，也没有提供类似于<torch/extension.h>这样的C++扩展库的头文件。

如果你需要在C++代码中使用Jittor，你可能需要直接使用Jittor的底层C++和CUDA代码。但是，这可能需要你对Jittor的内部实现有深入的了解，而且可能需要编写大量的额外代码。

如果你的代码需要在C++和Python之间进行交互，你可能需要使用一些工具，如SWIG或Cython，来创建一个C++和Python之间的接口。这可能需要一些额外的工作，但是它可以让你在C++代码中使用Jittor的功能。

总的来说，如果你的代码主要是在Python环境中运行，我建议你直接使用Jittor的Python API。如果你的代码需要在C++环境中运行，你可能需要使用Jittor的底层C++和CUDA代码，或者创建一个C++和Python之间的接口。


结论 **无法修改 因为光栅化器部分里面需要调用pytorch的c++ API 因为jittor没有C++ API 无法与cuda交互进行渲染，导致项目无法进行下去，因为无梯度的tensor无法进行反向传播** 

与jittor开发者交谈后 尝试用jtorch 替换torch与cuda交互以及保留中间梯度的功能

jtorch和jittor版本对应又出现了问题 一直过不去jtorch测试用例，先不管，先看看能不能用 目前测试版本为jittor1.3.7.3+jtorch0.7.0 都过不去 把jtorch代码改了放宽jittor版本限制也不行

与吴学长交谈后，提供了三种思路给我解决光栅化器部分的问题
1. 在将image转为numpy之前，先显式的存储梯度属性，用变量存储后再转为numpy 无法显示存储 因为返回的image在计算loss之前 没有backward也就无法显式保留梯度？但是或许可以尝试把他改成直接对image进行backward 然后保存其他中间变量的梯度？但是入口处也需要backward？感觉更复杂了 就要进行多次backward 整个计算图都碎片化了
2. 定义一个损失函数loss=a+bx，b就为此时的梯度，然后对损失函数想对于x求梯度就是梯度值本身
3. 按照光栅化器原理手写一个jittor/pytorch版本的光栅化器

思考并尝试后，觉得在一个项目框架中用两种深度学习框架很容易导致梯度的断裂，因为二者的计算图不同，且二者间不可以相互转换，只能以numpy为中介进行转换，而这样就会导致梯度在转为numpy的时候断裂，因为numpy不支持自动微分，所以梯度信息会被丢失，在光栅化器部分如果使用pytorch与cuda，需要先将jittor转为numpy再转为torch，这样梯度就断了，就是在出口和接口处保留梯度信息也不可行，因为梯度在backward后才被计算，这意味着当loss没backward的时候，梯度全是None，所以不好保存

考虑了一下后，感觉还是最好只用一个深度学习框架，就算提前在光栅化器的输入和输出的接口部分进行backward更新梯度并储存，这样的梯度也是不连续的，局部的，整个backward没有连成一条完整的线，会导致backward计算出来的梯度的更新方向是局部的不是全局最优的，给性能造成很大的影响

所以开始利用jittor复现光栅化器部分

### 利用jittor复现光栅化器

尝试参考代码[torch-splatting](https://github.com/hbb1/torch-splatting/tree/main)复现光栅化器，这段代码没提供对simple-Knn的复现，到时候遇到问题了再修改，先试试看这个代码能不能用，然后看看能不能用它来替代光栅化器那部分

暂时没跑，大致浏览了torch-splatting与render部分有关的代码，主要集中在gauss_render.py函数，其中的render函数与原项目的render函数接口类似，功能类似，尝试看能否通过修改接口直接复用这部分函数

现在在尝试接torch_splatting的renderer部分到源代码中，torch_splatting没考虑对screenspace_points的梯度的保留与计算，之后再添加，先确保二者能对应上

目前遇到的问题是不知道如何接[https://github.com/hbb1/torch-splatting/issues/3](https://github.com/hbb1/torch-splatting/issues/3)这个里面提到的相机焦距和渲染size的问题，已解决


### 显存爆炸 

目前最炸裂的问题是跑完两次循环渲染后，直接就爆显存了，测试似乎不是两层循环轮数的问题，将tilesize从64升到256，只有大概4*6次循环，显存仍然占用了11G

接下来的思路
1. 转换到高显卡GPU上测试，如3090，测试显存是否足够
2. 将输入数据集进行处理，选用图像size更小的数据集

现在主要考虑对输入的数据集进行处理以降低显存占用，有以下几点思路
1. 降低图像分辨率：直接降低每张图像的像素数量减少显存占用，在原项目中提供了命令行参数来调整输入图像的分辨率，--resolution / -r 默认为1，改为2 4 8 分别将图像分辨率降低到0.5 0.25 0.125 将分辨率调到直接调到256后 不再爆显存

通过利用jittor里的knn 删去了对子模块的所有依赖，现在的问题是仍然求不出梯度，梯度算出来全部是0

是有梯度的 比较异常的就是screenspace points还是没有梯度，且self.rotation没有梯度，但是很奇怪的是在原版代码中没有梯度的是self._features_rest,估计应该还是因为这个screenspacepoints梯度的不存在是因为jittor没有保存中间变量的原因，

最后返回means2D作为梯度，然后借助优化器求梯度，暂时可以跑通一个iter 但是到第二个iter就显存爆炸。。。。

经过调试发现在第二个iter进行渲染时，显存爆炸，第一次渲染一次占用的内存是6G左右

调来调去 感觉还是显存的问题 最后把图片的分辨率调到128 不在爆显存。。。 发现 把tilesize降低后 显存显著降低，最后搭配的组合是 -r 8 tilesize 16


### 稠密化部分
开始进行稠密化部分的修改 因为源代码中稠密化的iter比较高 为了测试 我们这里把默认设置中的iter都调小 --densify_from_iter 先调到2 --densification_interval 先调到1

这样调了之后mask似乎全是false 但是同样的命令行参数官方代码里面的mask不是false，感觉可能是优化器不同导致结果不完全一致的原因 或许需要先比较一下输入数据的区别以及尝试调大参数

应该是之前在渲染部分 渲染器用的不一样 这会使得求出来的梯度不一样 而在densifacion这一部分主要就是靠梯度累积和梯度累计次数两个参量作为依据更新点的位置 所以会有区别

算出来的梯度好像都太小了，根本不能拿来更新，g了 随着迭代次数的上升 梯度的累计并没有增加 推测是因为我每个迭代都把内存完全清理的原因。。

torchsplatting里面利用means2D.retain_grad保留得到的梯度 明显不是全为0的 感觉问题还是出在对梯度的计算不正确上面,而且每次迭代的梯度都不一样 现在JGaussia里面的梯度多次迭代的结果的数值都一样 应该是梯度计算有问题 梯度有点区别 但区别不大 很多都是0 有区别的量级也很小

torchsplatting里面的梯度测试后每个Gaussian点都有值 16384个点16384个梯度，但是源代码中54000多个点 只有14000个有值，JGaussian里面52000多个点 只有2858个有梯度 ------screenspace_points

JGaussian 2890/54000有梯度 torch_splatting 16384/16384 有梯度 GaussianSplatting 14058/54000 Gaussian_torch 15221第一次 28787第二次 


### 优化器参数梯度全为0

现在尝试直接将返回的viewpoint_tensor张量添加到优化器后再进行优化 先尝试把优化器的设置函数搬到渲染完之后 算出来的grad还是0。。

裂开了 发现问题了 服了 优化器参数梯度肯定算出来 这么多0 说明问题很大 后面优化器肯定很多Gaussian点都不会更新 还是得找出这个地方问题在哪， Gaussian 和torch_splatting里面的优化器参数的梯度都是很正常的 就我的很奇怪 应该是哪里有问题

现在先不管densification 先想办法把优化器参数梯度算对再说 算不对 什么都是错的

一步步排查梯度在哪有问题 现在预计问题出在gauss_render.py 感觉很有可能是因为我最后对tilesize的那个变形操作带来的影响 

查阅了一些资料发现在对要求梯度的变量中 最好不用inplace操作

in-place 操作可能会覆盖计算梯度所需的值。
**每个 in-place 操作实际上都需要重写计算图的实现。out-of-place只是分配新对象并保留对旧计算图的引用，而 in-place 操作则需要将所有输入的创建更改为代表此操作的函数。**

一步步排吧 改了好像也不是那的问题 目前确定的是问题出在render里面而不是后面的loss计算也不是在forward里 **大概率在render里**,因为render后返回的梯度很多都是0 在forward里面则不存在这种情况
 
感觉估计在渲染前有地方有问题 导致梯度断裂？不太可能 渲染前都是和源项目一步步对比数据来做的


发现torch-splatting里面一个很奇怪的点就是它的rotation的梯度都是0 

现在就xyz的梯度不全为0 但很奇怪的是 很多点的梯度都一模一样的

随着不断排除问题 现在认为应该是在render里出现了梯度消失 因为在forward里面每个参量的梯度都能和torchsplatting对应的上 现在需要思考并查阅资料看如何解决梯度消失，不是**梯度消失** 因为**梯度消失问题通常在与损失函数较远的代码部分发生，尤其是在深层神经网络中。这是因为在反向传播过程中，梯度通过多个层传递，逐渐减小。在离损失函数较远的层，梯度可能变得非常小，甚至趋近于零，从而导致权重无法得到有效的更新。**

进行多次迭代后还发现screenspace梯度一直都是一模一样 一点变化都没有

麻痹 找到问题了 我激活函数地方的torch代码没改 tmd  改了好像还是没啥用


需要详细看一下gauss_render函数 并分析一下自己是否把里面的函数都改对了,感觉没什么问题 对了一下shape都是对的 jittor和torch对应的函数也都没有功能性问题

有点难搞 现在就是52000个Gaussian点里面只有2000多个有梯度 然后每次迭代 xyz的坐标都似乎没有发生更新 也可能是更新太小？

不确定问题出在哪 先验证一下torch-splatting和Gaussian-splatting和在一起到底能不能用 如果可以 可能是jittor里面的问题 如果不可以，感觉还得自己写光栅化器

验证结果 torch_spltting的光栅化渲染器是可用的 与official代码结合后的梯度都是正确的 所以问题应该还是出在jittor搭建的JGaussian上面 而非torch_splatting的渲染器上面 

目前思路是逐步比对render部分JGaussian与Gaussian_torch之间的差异 找出哪个数据有问题 现在发现的是因为knn计算初始化的偏差 导致scales之间有区别 进而导致cov3d和cov2d有区别 不过差别不大 最后会导致render函数中 radii rect，dx,gauss_weight的计算有所区别 不过逻辑应该是对的 只是划分出来的矩形区域有点区别 导致Gaussian点的多少和位置有点变化，最后的color之前比torch版本大概大四倍左右

有梯度啦！！！！ 问题出在对pixel_coord的初始化不对，改正后有14000个点有梯度 和Gaussiansplatting 几乎一致 现在的问题转为优化器的问题 因为多次迭代后 数据没有更新

### 优化器参数只更新一次

找到问题 问题出在把优化器更新步骤放在了with jt.no_grad里面 jittor的no_grad会把优化器参数也变成0 而且不能再改回去

### densification

现在开始做densification

现在有个问题，就是Gaussian_splatting的densification其实在第3次迭代就由200多个点需要修剪 而Gaussian_torch和JGaussian都在第三次迭代中没有点要修剪 非常奇怪 可能是梯度阈值设置的问题 另外 GaussinSplatting第一次迭代就有150多个点需要修剪了 估计是梯度阈值的问题 现在开始对比梯度阈值 Gaussian——torch计算的梯度比Gaussian-splatting大约小2到4个量级，原来的梯度阈值为0.0002 现在先把他改为0.0000002 小三个量级 果然 修改后 Gaussian——torch由500多个点要修剪 而JGaussian有2000多个点要修剪 先用着 后续可能需要调参

在裁剪优化器参数时发现exp和exp_sq参数都不存在 后来发现是因为只有在运行过一次step后才会有exp和exp_sq

经过查阅资料和思考 Jittor中的Adam优化器参数与Pytorch有所不同 其内部状态参量是values m grads 这些需要仿照pytorch里面添加exp_avg和exp_avg_sq的方法添加并修改

似乎无法正确用键值对正确获取参数状态 因为jittor和pytorch的Adam优化器的形式不同 根据jittor优化器组织格式 采用索引来找值 我特么有点哈比了 可以不调用stact_dict方法 直接self.optimizer.default就可以得到属性 然后再用assign修改参数状态 但是很奇怪的是他好像没有更新优化器里的状态只有assign之后的值好像是对的 但是assign应该是不返回值的 很奇怪 但是Gaussian里面的也没有更新优化器 先往后面再走 看看会不会报错 


default 得出的属性无法用assign更新 感觉还是要尝试调用stact_dict方法,stact_dict方法就是范围default属性 没什么区别

default是只读函数 无法更新 对应的state_dict函数也是调用的只读函数default，最后选取的是优化器的param_group属性，并用assign修改数据,好像这种做法有些问题 因为 最后assign一次后 原来获取的values都是list assign一次后直接变成张量了，明天再改

这个地方有问题 assign之后导致values m, grads 都不再是list了 应该是stored_state与原优化器参数共享内存了 直接在前面已经修改好了 不用assign了

多次step报错 因为screenspcepoints的params形状发生了变化，所以决定还是在修改优化器参数时添加screenspacepoints 因为不加的话 在迭代求梯度时 用到的assign会把screenspacepoints的type更新为xyz的type[:, :2]

发现一个问题 screenspacepoint的梯度只更新了一次后就不动了,但是其他的优化器参数梯度都一直在更新,而且有个问题就是screenspacepoint的点的shape并没有变化？

修改后可以变化了 但是梯度还是没动 还是没有什么变化？ 嗯？ 好像所有梯度都没什么变化？
