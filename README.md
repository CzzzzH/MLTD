## Temporally Stable Metropolis Light Transport Denoising using Recurrent Transformer Blocks

Official Implementation for *Temporally Stable Metropolis Light Transport Denoising using Recurrent Transformer Blocks*

[Project Page](https://czzzzh.github.io/MLTD/) | [Paper Link](https://czzzzh.github.io/MLTD/MLTD.pdf)

![teaser](./teaser/teaser.png)

### Updates

[07/31/2024]. We released our core codes, pretrained model and part of the test dataset

 

### Quick Start (Linux only)

We recommend using `nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04` if you are to run our code in a docker image. We provide a quick start tutorial for denoising the 60-frame *Monkey* animation in the teaser. Proceed as follows:

First, simply create a virtual environment with **conda**:

```bash
conda create -n MLTD python=3.9 # Python3.9 is required for the code
```

Then use **pip** to install the dependencies (they are a bit redundant now and would be simplified then)

```bash
pip -r requirements.txt
```

Download our **pretrained model** and **test dataset** from Google Drive (you may only download `dataset/test_monkey_128spp_split`) and put them into `checkpoints` and `dataset` 

https://drive.google.com/drive/folders/1GvoSIQ9svfWToK4O4k7XLA1fFceTgDky?usp=sharing

Run the test command to denoise the *Monkey* with the default config `config/MLTD.yaml`:

```bash
python test.py \
       task_id=0 \
       num_workers=0 \
       test_dir="dataset/test_monkey_128spp_split" \
       statistics_dir="statistics/test" \
       split=True
```

The denoised frames will be saved at the `visualization`  directory. Feel free to check it!

You can also output the **Noisy Input** and the **Reference** by re-running the previous command with an extra cli parameter `test_input=True`.  

#### Notes:

(1) We use efficient kernel operator from [SBMC](https://github.com/adobe/sbmc). We have compiled the operator in Python3.9 in this repository (See `models/halide_ops.cpython-39-x86_64-linux-gnu.so`). If you have problem in using this operator, please compile the operator manually following [this repo](https://github.com/CzzzzH/Denoiser). 

(2) Check  `script/test.sh`  if you want to run other test cases

(3) We have re-trained our denoiser with an updated train setting and more training sequences. It is more powerful now!



### Dataset Generation

Before training the model by yourself, you may want to have the full dataset at first. Our full dataset is too large to release at Google Drive (we may release a subset of it later), but since our dataset is based on the open-source [OpenRooms](https://github.com/ViLab-UCSD/OpenRooms), it won't be difficult for you to recreate it.

Here we provide a **step-by-step** tutorial for recreating the training and validation dataset:

1. Download the scene configuration xml files (with camera poses), environment maps, materials, furniture and layout geometry of OpenRooms from [here](https://mclab.ucsd.edu/OpenRooms_public/). Put the directories into `scenes/OpenRooms`.  The folder structure should be

   ```bash
   scenes
   |-- OpenRooms
       |-- BRDFOriginDataset
       |-- EnvDataset
       |-- layoutMesh
       |-- scenes
       |-- uv_mapped
   ```

   Note that the **BRDFOriginDataset** is not released on the given server since the SVBRDF materials are from [Adobe Stock](https://stock.adobe.com/search?filters%5Bcontent_type%3A3d%5D=1&filters%5B3d_type_id%5D%5B0%5D=3&load_type=3d+lp) and need to be purchased. You may follow the [Tutorial](https://drive.google.com/file/d/1d751UulbaCMqo0cKo_HDPU2UyVKWWwP1/view) of OpenRooms to get a free substitution.

2. Compile our modified **Mitsuba0.6** renderer in `mitsuba`. [Here](https://medium.com/@sree_here/10-steps-to-install-mitsuba-renderer-on-ubuntu-38a9318fbcdf) is one helpful tutorial about how to install the dependencies and compiled on *Ubuntu*

   Some required libraries include:

   ```bash
   sudo apt-get install build-essential scons libglu1 libpng-dev libjpeg-dev libilmbase-dev libxerces-c-dev libboost-all-dev libopenexr-dev libglewmx-dev libxxf86vm-dev libeigen3-dev libfftw3-dev
   ```

   If you don't have our modified Mitsuba renderer , please re-clone the repository with:

   ```bash
   git clone https://github.com/CzzzzH/MLTD.git --recursive
   ```

3. Run the script to generate the dataset:

   ```bash
   bash scenes/generate_train_valid.sh
   ```

   This script only generates one training sequence and validation sequence for example. Remove or edit the `--end-idx` and `--camera-idx` arguments to customize your generation. 

#### Notes:

(1) We have created the training & validation data list in `scenes/indoors/valid_renderings_indoors_train.json` and `scenes/indoors/valid_renderings_indoors_valid.json`, which removes the invalid data in the automatically generated scene configurations with some conditions. See the Python script under `scenes` to find out how we obtain the data list and customize it for your need. 

(2) Though supporting MLT and PSSMLT,  **Mitsuba0.6** only provides CPU rendering for them, which is very slow. We recommend using a lower sample count to render the reference at the first step. According the theory from [Noise2Noise](https://arxiv.org/pdf/1803.04189), you can still train a good denoiser without high-quality references (Our newest model was trained with references only rendered with **1024spp**)

(3) Our modified Mitsuba renderer only supports the sample decomposition for **MLT** and **PSSMLT** now. 



### Training

If you have generated data in `dataset/train_indoors` and  `dataset/valid_indoors`, then simply run the following script to train the model with the standard setting:

```bash
python train.py --config configs/MLTD.yaml
```

Check `configs/MLTD.yaml` for more training options.

#### Note:

Our model can be generalized to denoising other rendering algorithm like **path tracing** or **bidirectional path tracing**. Though we don't have any pretrained model for them, you are welcome to train them by yourself with our dataset or your own dataset. 

In this repository, we provide a canonical model **without sample decomposition** that enables you to train a denoiser for arbitrary rendering algorithm.

You can use the MLT dataset for a try:

```bash
python train.py --config configs/MLTDSimple.yaml
```



### Acknowledgement

We acknowledge the following repositories for borrowing the codes:

**Transformer Block:** https://github.com/swz30/Restormer

**Kernel Operator:** https://github.com/adobe/sbmc

**Patch Sampling:** https://github.com/Nidjo123/kpcn



### Citation	

If you find this repository useful in your project, welcome to cite our work :)

```
@article{Chen:2024:MLTD,
  author = {Chen, Chuhao and He, Yuze and Li, Tzu-Mao},
  title = {Temporally Stable Metropolis Light Transport Denoising using Recurrent Transformer Blocks},
  journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH)},
  year = {2024},
  volumn = {43},
  number = {4},
  articleno = {123}
}
```

