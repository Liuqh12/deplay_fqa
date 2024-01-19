# Libtorch 篇

C++版本的PyTorch，也可以用来部署模型。Tips: ***改一个地方后，一定要记得跑一遍推理***。

## 使用 torch.jit.trace 还是 torch.jit.script?
总体结论：forward中不包含基于分支的数据处理，用trace，否则用script。限制条件有点严格。

## torch.jit.trace 坑
### 能save，能推理，未必结果正确
导出方法：
```python
gfapgan = module()
gfpgan.eval()
sample_input = torch.randn(size=(1, 3, 512, 512), dtype=torch.float32)
m = torch.jit.trace(gfpgan, sample_input)
torch.jit.save(m, 'trace.pt')
```

推理方法：
```python
img = cv2.imread("inputs\cropped_faces\Justin_Timberlake_crop.png")

cropped_face_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
cropped_face_t = cropped_face_t.unsqueeze(0).to('cuda')

model = torch.jit.load("trace.pt").to('cuda')

output = model(cropped_face_t)[0]
restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
cv2.imshow("new img", restored_face)
```

解决方法：导出后，先用python验证下。

## torch.jit.script 
### Runtime error:Unsupported value kind: Tensor
一些类似问题：
1. [issues](https://github.com/pytorch/pytorch/issues/107568)
2. [论坛](https://discuss.pytorch.org/t/runtime-error-unsupported-value-kind-tensor/190276)

报错：
```text
Traceback (most recent call last):
  File "d:\lqh12\GFPGAN-1.3.8\GFPGAN-1.3.8\test_export_libtorch_pt.py", line 46, in <module>
    m = torch.jit.script(ONNX_Model())
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_script.py", line 1324, in script
    return torch.jit._recursive.create_script_module(
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_recursive.py", line 559, in create_script_module
    return create_script_module_impl(nn_module, concrete_type, stubs_fn)
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_recursive.py", line 632, in create_script_module_impl
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_script.py", line 639, in _construct
    init_fn(script_module)
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_recursive.py", line 608, in init_fn
    scripted = create_script_module_impl(
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_recursive.py", line 632, in create_script_module_impl
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_script.py", line 639, in _construct
    init_fn(script_module)
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_recursive.py", line 608, in init_fn
    scripted = create_script_module_impl(
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_recursive.py", line 636, in create_script_module_impl
    create_methods_and_properties_from_stubs(
  File "C:\Users\10-20\anaconda3\envs\gfpgan\lib\site-packages\torch\jit\_recursive.py", line 469, in create_methods_and_properties_from_stubs
    concrete_type._create_methods_and_properties(
RuntimeError: Unsupported value kind: Tensor
```

问题根源：forward() 需要三个参数，调用时传入两个，剩下一个使用默认值。
示例：
```python
class ToRGB(nn.Module):
    def __init__(self, in_channels, num_style_feat, upsample=True):
        ...
    def forward(self, x, style, skip=None):
        ...
        if skip is not None:
            do ...
        return out

l = ToRGB()

x = l(x, style)         # 不可以

x = l(x, style, skip)   # 可以    
```

解决方法：每个调用forward的地方都传递三个函数。

Tips：skip通常是一个Tensor，所以推荐生成一个空Tensor代替None传入。如下：
```python
empty_skip = torch.empty(size=(0, 1))
```
在forward中更改判断如下：
```python
def forward(self, x, style, skip):
    ...
    if skip.numel() != 0:
        do ...
    return out
```

### Did you forget to initialize an attribute in __init__()?:
报错：
```text
RuntimeError:
'__torch__.gfpgan.archs.stylegan2_clean_arch.___torch_mangle_31.StyleConv (of Python compilation unit at: 000001CEF85275B0)' object has no attribute or method '__call__'. Did you forget to initialize an attribute in __init__()?:
  File "d:\lqh12\GFPGAN-Clean\gfpgan\archs\gfpganv1_clean_arch.py", line 103
        i = 1
        for conv1, conv2,  to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], self.to_rgbs):
            out = conv1(out, latent[:, i])
                  ~~~~~ <--- HERE
            # the conditions may have fewer levels
            if i < len(conditions):
```
self.style_convs对应初始化：
```python
def __init__():
    ...
    self.style_convs = nn.ModuleList()
    ...
    for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode='upsample'))
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None))
```
StyleConv:
```python
class StyleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, demodulate=True, sample_mode=None):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, num_style_feat, demodulate=demodulate, sample_mode=sample_mode)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x, style):        
        out = self.modulated_conv(x, style) * 2**0.5  # for conversion        
        b, _, h, w = out.shape
        noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise        
        out = out + self.bias
        out = self.activate(out)
        return out
```

```__call__```让一个Object可以通过```Object(input)```的方式被调用。也就是说：在这里，实现```__call__```是能默认调用forward的前提，这不是排查重点（自己写个call依旧报错)。
确保每一个变量都被初始化过，而不是在某种情况下才初始化。forward中，局部变量```noise```根据```out```的shape生成，很难搞。Issue搜索结果：[ModuleList ](https://github.com/pytorch/pytorch/issues/47336)，貌似是序列问题。改一下代码，手动调用forward：
```python
out = conv1.forward(out, latent[:, i])
```
### ModuleList/Sequential indexing is only supported with integer literals.
报错：
```text
RuntimeError:
Expected integer literal for index but got a variable or non-integer. ModuleList/Sequential indexing is only supported with integer literals. For example, 'i = 4; self.layers[i](x)' will fail because i is not a literal. Enumeration is supported, e.g. 'for index, v in enumerate(self): out = v(inp)':
  File "d:\lqh12\GFPGAN-Clean\gfpgan\archs\gfpganv1_clean_arch.py", line 294
        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
                   ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
            unet_skips.insert(0, feat)
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)
```
提示信息很完整。我用的版本是torch：2.1.2+cu121，根据[issue](https://github.com/pytorch/pytorch/pull/45716)，貌似已经被解决，让人疑惑。前后对比代码:
```python
        eee = int(self.log_size - 2)

        for i, l in enumerate(self.conv_body_down):
            if i < eee:
                feat = l(feat)
                unet_skips.insert(0, feat)
        # 原始代码
        # for i in range(self.log_size - 2):
        #     feat = self.conv_body_down[i](feat)
        #     unet_skips.insert(0, feat)
```
如果是多个ModuleList嵌套调用，可以用zip压缩后，使用上述方法：
```python
m_s = zip(self.conv_body_up, self.condition_scale, self.condition_shift)
for i, (l, ll, lll) in enumerate(m_s):
    if i < eee:
        feat = feat + unet_skips[i]
        feat = l(feat)
        scale = ll(feat)
        conditions.append(scale.clone())
        shift = lll(feat)
        conditions.append(shift.clone())

# 原始代码
# for i in range(self.log_size - 2):
#     # add unet skip
#     feat = feat + unet_skips[i]
#     # ResUpLayer
#     feat = self.conv_body_up[i](feat)
#     # generate scale and shift for SFT layers
#     scale = self.condition_scale[i](feat)
#     conditions.append(scale.clone())
#     shift = self.condition_shift[i](feat)
#     conditions.append(shift.clone())
```
深入分析相关issues发现，ModuleList和Sequential在script中处理流程并不一致。前的的问题中提到了需要手动调用forward的问题，从这里也算得到验证。综合来说，ModuleList灵活性更高，可以定制forward（炼丹者的套娃，部署人的灾难）。Python里面的迭代属实老生常谈。
