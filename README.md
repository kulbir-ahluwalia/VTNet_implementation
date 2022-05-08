# VTNet (UIUC CS-444 Project)
Implementation of VTNet ([Paper](https://arxiv.org/pdf/2105.09447.pdf)).

The code is largely based on [the original code](https://github.com/xiaobaishu0097/ICLR_VTNet) provided by the author of the paper. A lot of classes are directly picked from the original repository. This repository only supports DETR features. It also provides an option to use `nn.Transformer` in place of the `VisualTransformer` provided with the original code. There is a difference how local and global features are passed to `nn.Transformer` and `VisualTransformer`. Check the code for details.
