# VTNet: Visual Transformer Network for Object Goal Navigation

Make the conda environment:
```angular2html
conda create --name VTNet_test_env  python=3.6 
conda activate VTNet_test_env  

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia    
conda install -c conda-forge ipywidgets


conda install ipykernel
conda install ipywidgets
python3 -m pip install --upgrade --user urllib3==1.25.9
python3 -m pip install --upgrade --user pillow==6.2.0
conda install matplotlib   
conda install -c anaconda scikit-learn
pip install opencv-python
pip install urllib3

pip install -U torch #??
pip install --upgrade protobuf

```
Delete pytorch from requirements.txt.


## Installation
The code is tested with Ubuntu18.04 and CUDA10.2.
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



## Testing

```python
python full_eval.py --gpu-ids 0 --detr --save-model-dir {SAVE_MODEL_DIR} --results-json ./result.json --model VTNetModel --title a3c_previstrans_base
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
