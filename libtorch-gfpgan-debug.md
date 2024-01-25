# 基于Libtorch的GFPGAN推理Debug
千辛万苦torch.jit.script(gfpgan)不报错，推理结果却匪夷所思：

原图如下：
<br>
<img src="./imgs/source.png" width="25%" title="原图">

预期如下：
<br>
<img src="./imgs/pytorch.png" width="25%" title="PyTorch推理结果">

实际如下：
<br>
<img src="./imgs/libtorch.png" width="25%" title="LibTorch推理结果">

# 现象

以相机作为数据源，实时输出推理结果，一直显示错误的图像。而且，错误图像并不随真实图像中内容变化有明显变化。

# 确定排查方向

以下信息是确定的：
1. 输入到模型中的待计算数据是相同的。
2. 获取到的模型推理结果是正确的。
3. 没有任何涉及到script导出相关的Warning。

现有怀疑点如下：
1. 导出错误。导出过程没有错误，只代表没有编译错误，不代表导出的pt文件可用。GFPGAN[仓库](https://github.com/TencentARC/GFPGAN/tree/master)中原始代码中有若干条件判断，从结构上讲有一定的复杂性。
2. pt文件推理过程中某个（些）节点计算错误。

# 初步排查
## 查找DeBug工具
原则上讲，script作为python的子集，应该也可以使用pdb类的工具。
