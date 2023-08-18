## Neumann Network with Recursive Kernels for Single Image Defocus Deblurring<br><sub>Official PyTorch Implementation of the CVPR 2023 Paper</sub><br><sub>
This repo contains training and evaluation code for the following paper:

> [**Neumann Network with Recursive Kernels for Single Image Defocus Deblurring.**](https://openaccess.thecvf.com/content/CVPR2023/html/Quan_Neumann_Network_With_Recursive_Kernels_for_Single_Image_Defocus_Deblurring_CVPR_2023_paper.html)

> *IEEE Computer Vision and Pattern Recognition (**CVPR**) 2023*


## Getting Started
### Prerequisites

```shell
conda create -n NRKNet python=3.7
conda activate NRKNet
cd ./NRKNet
pip install -r requirements.txt
```


###  Datasets
Download and unzip datasets under `[DATASET_ROOT]`:
* DPDD dataset: [Google Drive](https://drive.google.com/open?id=1Mq7WtYMo9mRsJ6I6ccXdY1JJQvwBuMuQ&authuser=codeslake%40gmail.com&usp=drive_fs) | [Dropbox](https://www.dropbox.com/s/w9urn5m4mzllrwu/DPDD.zip?dl=1)
* CUHK test set: [Google Drive](https://drive.google.com/open?id=1Mol1GV-1NNoSX-BCRTE09Sins8LMVRyl&authuser=codeslake%40gmail.com&usp=drive_fs) | [Dropbox](https://www.dropbox.com/s/zxjhzuxsxh4v0cv/CUHK.zip?dl=1)
* RealDOF test set: [Google Drive](https://drive.google.com/open?id=1MyizebyGPzK-VeV1pKVf7OTDl_3GmkdQ&authuser=codeslake%40gmail.com&usp=drive_fs) | [Dropbox](https://www.dropbox.com/s/arox1aixvg67fw5/RealDOF.zip?dl=1)
* LFDOF dataset: [Download path](https://sweb.cityu.edu.hk/miullam/AIFNET/dataset/LFDOF.zip)
* RTF test set: Please contact the author (Laurent, laurentdandres@gmail.com), he will kindly share the dataset and necessary information with you.
```
[DATASET_ROOT]
 ├── DPDD
 ├── RealDOF
 ├── CUHK
 ├── LFDOF
 └── RTF
 
```
> `[DATASET_ROOT]` can be modified with [`config.data_offset`] in `./config.py`.

## Train the NRKNet

Train the NRKNet with different training datasets (DPDD | LFDOF).
```shell
# Trained with DPDD
CUDA_VISIBLE_DEVICES=0 python train_DPDD.py

# Trained with LFDOF
CUDA_VISIBLE_DEVICES=0 python train_LFDOF.py
```
## Test the NRKNet
Test the pre-trained models for CVPR.

#### Options
* Select the training and testing datasets in config.py. 
  * 'train['train_dataset_name']':  The name of a dataset to train. `DPDD` | `LFDOF`. Default: `DPDD`
  * 'test['dataset']':  The name of a dataset to evaluate. `DPDD` | `LFDOF` | `RTF`| `RealDOF`. Default: `DPDD`
   
* Run test.py. 
```shell
CUDA_VISIBLE_DEVICES=0 python test.py
```

#### Test with your re-trained models
* Modify the path of a re-trained model in config.py.

```shell
# From
train['resume'] = './save/NRKNet_' + train['train_dataset_name'] + '/0'

#To
train['resume'] = './save/NRKNet_' + train['train_dataset_name'] + '/1'
```

* Select the training and testing datasets in config.py.

* Run test.py

## Contact
Open an issue for any inquiries.
You may also have contact with [zicongwu.scut@gmail.com](zicongwu.scut@gmail.com)

## Citation

```
@inproceedings{quan2023neumann,
  title={Neumann Network With Recursive Kernels for Single Image Defocus Deblurring},
  author={Yuhui Quan, Zicong Wu and Hui Ji},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5754--5763},
  year={2023}
}
```
