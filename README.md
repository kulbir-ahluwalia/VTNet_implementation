# VTNet (UIUC CS-444 Project)
Implementation of VTNet ([Paper](https://arxiv.org/pdf/2105.09447.pdf)).

The code is largely based on [the original code](https://github.com/xiaobaishu0097/ICLR_VTNet) provided by the author of the paper. A lot of classes are directly picked from the original repository. This repository only supports DETR features. It also provides an option to use `nn.Transformer` in place of the `VisualTransformer` provided with the original code. There is a difference how local and global features are passed to `nn.Transformer` and `VisualTransformer`. Check the code for details.

## Running the code
Run the following for pretraining:
```bash
python pretraining.py \
                      --data-dir /ssd2/CS444_project/AI2Thor_VisTrans_Pretrain_Data \
                      --out-dir /ssd2/CS444_project/pretraining_output \
                      --batch-size 32 \
                      --num-workers 16 \
                      --epochs 30 \
                      --do-test \
                      --save-every 5 \
                      --use-nn-transformer
```

Run the following for training:

If running for the first time you can comment out ```assert args.use_nn_transformer == saved["args"].use_nn_transformer``` in training.py.
Use the following command for training with an expected RAM usage of 16 GB:
```bash
python  training.py  \                                                                                                                                                                                                         ─╯
                   --data-dir /ssd2/CS444_project/AI2Thor_offline_data_2.0.2 \
                   --out-dir /ssd2/CS444_project/training_output \
                   --workers 2 \
                   --max-ep 1000 \
                   --save-every 100 \
                   --pretrained-vtnet /ssd2/CS444_project/vtnet_pretrained_checkpoint.pth \
                   --verbose \
                   --num-workers 2

```

Once you have trained for some epochs,you can load your init model from /training_output: 
```bash
python  training.py  \
                   --data-dir /ssd2/CS444_project/AI2Thor_offline_data_2.0.2 \
                   --out-dir /ssd2/CS444_project/training_output \
                   --workers 2 \
                   --max-ep 1000 \
                   --save-every 1000 \
                   --use-nn-transformer \
                   --pretrained-vtnet /ssd2/CS444_project/vtnet_pretrained_checkpoint.pth \
                   --verbose \
                   --init-model INIT_MODEL \
                   --num-workers 2

```

# Using ACKTR instead of A3C

## Environment setup

requirements.txt has:
```
gym
matplotlib
pybullet
stable-baselines3
h5py
```

Setup environment:
```
conda create --clone VTNet_test_env --name ACKTR_pytorch_test_env 
conda activate ACKTR_pytorch_test_env
pip install -r requirements.txt  
conda install -c conda-forge gym-atari
```

Running the acktr.py only model on Pong:
```
 python acktr.py --env-name "PongNoFrameskip-v4" --num-processes 1 --num-steps 20 --save-dir ./trained_models/   
```


