# VTNet: Visual Transformer Network for Object Goal Navigation

Make the conda environment:
```angular2html
conda create --name VTNet_test_env  python=3.6 
conda activate VTNet_test_env  

#install pytorch and other required packages
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia    
conda install -c conda-forge ipywidgets
conda install ipykernel ipywidgets matplotlib 
conda install -c conda-forge setproctitle


python3 -m pip install --upgrade --user urllib3==1.25.9
python3 -m pip install --upgrade --user pillow==6.2.0 
conda install -c anaconda scikit-learn
pip install opencv-python urllib3 

#the following command resolves some errors
pip install -U torch #??
pip install --upgrade protobuf

```
Delete pytorch==1.4.0 from requirements.txt because the RTX 3060 on Ubuntu 18 needs the following in requirements.txt:
```
torch==1.10.2
torchvision==0.11.2
torchaudio==0.10.2+cu113
tensorboardX==2.5.0
```


## Installation
The code is tested with Ubuntu18.04 and CUDA10.2 on the campus cluster. For the RTX 3060 local setup used for fast development, we have cudatoolkit      11.1.74.
```
pip install -r requirements.txt
```


## Training

Before pre-training the VT, you could download the dataset [here](https://drive.google.com/file/d/1dFQV10i4IixaSUxN2Dtc6EGEayr661ce/view?usp=sharing).

### Pre-training
```
python main_pretraining.py --gpu-ids 0 --workers 4 --model PreTrainedVisualTransformer --detr --title a3c --work-dir ./work_dirs/
```

The training dataset could be downloaded [here](https://drive.google.com/file/d/1kvYvutjqc6SLEO65yQjo8AuU85voT5sC/view?usp=sharing) and the link of DETR features is [here](https://drive.google.com/file/d/1d761VxrwctupzOat4qxsLCm5ndC4wA-M/view?usp=sharing).


### A3C training 
```python
python main.py --gpu-ids 0 --workers 4 --model VTNetModel --detr --title a3c_vtnet --work-dir ./work_dirs/
```

For the local setup using RTX 3060:

```
python main.py --gpu-ids 0 --workers 1 --model VTNetModel --detr --title a3c_vtnet --work-dir ./work_dirs/ --pretrained-trans /ssd2/VTNet_implementation/vtnet_pretrained_checkpoint.pth --batch-size 64 --ep-save-freq 10000 --epochs 20

```



## Testing

```python
python full_eval.py --gpu-ids 0 --detr --save-model-dir {SAVE_MODEL_DIR} --results-json ./result.json --model VTNetModel --title a3c_previstrans_base
```

For local setup on RTX 3060:

We have to mention the {path to the saved model directory} and the path where we want the result.json to be saved.
```
python full_eval.py --gpu-ids 0 --detr --save-model-dir /ssd2/VTNet_implementation/work_dirs/a3c_vtnet_train_2022-04-12_13-11-25/trained_models/  --results-json /ssd2/VTNet_implementation/work_dirs/a3c_vtnet_train_2022-04-12_13-11-25/trained_models/result.json --model VTNetModel --title a3c_vtnet --batch-size 1

```
Since the dataset is about 80GB in size, it is recommended to download it on a external storage disk and then extract it on local machine using CLI. GUI hangs most of the time due to the sheer size of the dataset and you can get "no space left on device error" even if there is space left.
```
tar -xvf /media/kulbir/SSD_storage/CS444_project_ssd/AI2Thor_offline_data_2.0.2.tar.gz -C /ssd2/CS444_project
tar -xvf /media/kulbir/SSD_storage/CS444_project_ssd/AI2Thor_offline_data_2.0.2_detr_features.tar.gz -C /ssd2/CS444_project 
tar -xvf /media/kulbir/SSD_storage/CS444_project_ssd/AI2Thor_VisTrans_Pretrain_Data.tar.gz -C /ssd2/CS444_project 
```






# REMOTE SETUP ON UIUC CAMPUS CLUSTER
Transfer the files from your local computer to the campus cluster scratch directory using scp:
```
scp /ssd2/VTNet_implementation/AI2Thor_offline_data_2.0.2.tar.gz ksa5@cc-login.campuscluster.illinois.edu:/home/ksa5/scratch/VTNet_data 

scp /ssd2/VTNet_implementation/AI2Thor_offline_data_2.0.2_detr_features.tar.gz ksa5@cc-login.campuscluster.illinois.edu:/home/ksa5/scratch/VTNet_data 

scp /ssd2/VTNet_implementation/AI2Thor_VisTrans_Pretrain_Data.tar.gz ksa5@cc-login.campuscluster.illinois.edu:/home/ksa5/scratch/VTNet_data 

#transfer the pretrained weights file to remote repo on cluster:
scp /ssd2/VTNet_implementation/vtnet_pretrained_checkpoint.pth ksa5@cc-login.campuscluster.illinois.edu:/home/ksa5/VTNet_implementation

```
Extract the files:
-x = extract, -v = display the extracted file in terminal, Use the --directory (-C) to extract archive files in a specific directory
```
tar -xvf AI2Thor_VisTrans_Pretrain_Data.tar.gz -C ~/scratch/VTNet_data/
tar -xvf AI2Thor_offline_data_2.0.2.tar.gz -C ~/scratch/VTNet_data/
tar -xvf AI2Thor_offline_data_2.0.2_detr_features.tar.gz -C ~/scratch/VTNet_data/

```

## Changing the file paths
Change the file paths in full_eval.py, main.py and main_pretraining.py
```
    args.data_dir = os.path.expanduser('/home/ksa5/scratch/VTNet_data/AI2Thor_offline_data_2.0.2/')
    args.data_dir = os.path.expanduser('/home/ksa5/scratch/VTNet_data/AI2Thor_offline_data_2.0.2/')
    args.data_dir = os.path.expanduser('/home/ksa5/scratch/VTNet_data/AI2Thor_VisTrans_Pretrain_Data/')
```

## Switch to UIUC_campus_cluster_VTNet branch
```
git checkout UIUC_campus_cluster_VTNet
```

## Using vim
Using vim:
```
i: to enter insert mode
:w ==> to write(save) but not exit
:q ==> to quit(exit)
:wq ==> to write and quit
ESC ==> to escape insert mdoe
```
Change your ~/.bashrc file and include the following and then save it:
```
module load python/3

export PYTHONPATH=/home/$USER/scratch/mypython3:${PYTHONPATH}

module load anaconda/2019-Oct/3

conda activate VTNet_test_env

```
After saving .bashrc, do: ```source ~/.bashrc```.

## Setup conda environment and ~/.bashrc





# LOCAL SETUP on RTX 3060
## Changing the file paths
Change the file paths in full_eval.py, main.py and main_pretraining.py
```
    args.data_dir = os.path.expanduser('/ssd2/CS444_project/AI2Thor_offline_data_2.0.2/')
    args.data_dir = os.path.expanduser('/ssd2/CS444_project/AI2Thor_offline_data_2.0.2/')
    args.data_dir = os.path.expanduser('/ssd2/CS444_project/AI2Thor_VisTrans_Pretrain_Data/')


```

# Errors and their solutions for local setup using RTX 3060

## Error 1
Error: RuntimeError: cuda runtime error (3) : initialization error at /opt/conda/conda-bld/pytorch_1544174967633/work/aten/src/THC/THCGeneral.cpp:51
Solution: https://github.com/pyg-team/pytorch_geometric/issues/131 

Add the following:
```
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
```

## Error 2
Error: RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED using pytorch
Solution: https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch
Call the following function:
```
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

```

## Error 3
Error: Path or directory not found
Solution: change the path to your training datasets and change the paths which are valid for your setup.
For example in offline_controller_with_small_rotation.py:
```
offline_data_dir='/ssd2/CS444_project/AI2Thor_offline_data_2.0.2',
```

Also, in full_eval.py:
```
args.data_dir = os.path.expanduser('/ssd2/CS444_project/AI2Thor_offline_data_2.0.2/')
```

In main.py:
```
args.data_dir = os.path.expanduser('/ssd2/CS444_project/AI2Thor_offline_data_2.0.2/')
```

## Error 4
Error: CUDA out of memory in training
Solutions: 
1. Use --workers 1 #and not --workers 4
2. Use --batch-size 64 or --batch-size 32

```
python main.py --gpu-ids 0 --workers 1 --model VTNetModel --detr --title a3c_vtnet --work-dir ./work_dirs/ --pretrained-trans /ssd2/VTNet_implementation/vtnet_pretrained_checkpoint.pth --batch-size 64 --epochs 20

```

## Error 5
Error: Disk out of space
Solution: Save the model weights after 10,000 episodes only
USe the argument: --ep-save-freq 10000

```
 python main.py --gpu-ids 0 --workers 1 --model VTNetModel --detr --title a3c_vtnet --work-dir ./work_dirs/ --pretrained-trans /ssd2/VTNet_implementation/vtnet_pretrained_checkpoint.pth --batch-size 64 --ep-save-freq 10000 --epochs 20

```

## Error 6
Warning: 
```
/ssd2/VTNet_implementation/models/vtnetmodel.py:226: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)

```
Solution:
Update the following lines in vtnetmodel.py:
```
floor_division = torch.div(dim_t, 2, rounding_mode='floor')
# dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)
dim_t = 10000 ** (2 * (floor_division) / c_pos_embedding)
```
    
# Using the campus cluster

## Using SSH
We use the following format to login to the head node:
```
ssh -X -l netid cc-login.campuscluster.illinois.edu   
```
Submit the job on the campus cluster using the following command:
```
qsub campus_cluster_script.pbs
```


## Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{
    du2021vtnet,
    title={{\{}VTN{\}}et: Visual Transformer Network for Object Goal Navigation},
    author={Heming Du and Xin Yu and Liang Zheng},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=DILxQP08O3B}
}
```
