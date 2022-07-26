# HouseX

HouseX is a fine-grained house music dataset, including 160 tracks, that provides 4 sub-genre labels and around 17480 converted mel-spectrograms of slices of these tracks. Besides, this repository includes several baseline models for classification on these mel-spectrograms. Additionally, the start and the end of the drop of each track is annotated, as provided by the `.json` files. The four `.xlsx` files contain metadata of the songs we selected, including the key, the alternate key and the BPM.

## Dataset
The original paper introducing this dataset can be accessed on [Arxiv](https://arxiv.org/abs/2207.11690). The mel-spectrograms are available at 
[HouseX processed data (Google Drive link)](https://drive.google.com/drive/u/1/folders/1HHi_WadYdea791zOq0Ib07AAPsR__yH-).

P.S. The paper has been accepted by [APSIPA ASC 2022](https://www.apsipa2022.org/).


## Environment Setup
We recommend using conda virtual environment. First, run

```
conda create -n HouseX python=3.8
conda activate HouseX
```

to create and activate the environment.

For PyTorch and Torchvision, you can install the version that matches your system following [the official installation guide](https://pytorch.org/get-started/locally/). We used a previous version for convenience:

`pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

Then, run

`pip install -r requirements.txt`

to install other required packages.

You may also need to install `ffmpeg` using the following command:

`conda install ffmpeg -c conda-forge`


## Training

The project file tree should be structured as follows:
```
📦HouseX
 ┣ 📜infer.py
 ┣ 📜logger.py
 ┣ 📜plot.py
 ┗ 📜train.py
 ┣ 📂logs
 ┣ 📂param
 ┣ 📂melspecgrams
 ┃ ┣ 📂test
 ┃ ┣ 📂train
 ┃ ┣ 📂val
```
The `./melspecgrams` directory could be derived either from `melspecgrams_all.zip` or `melspecgrams_drop.zip` provided in the Google Drive link.
Since the computing tasks are done on the NYU HPC cluster, we need to manually load the state dictionary of the pretrained torchvision models. If you want to replicate the experiments, simply delete the line that load state dictionary in `train.py` and set `pretrained = True`. For example, if you want to use pretrained ResNet18, set `backbone = models.resnet18(pretrained=True)`.

To train a model, run the following command:

`python train.py --id {TASK_ID} --do_train --pretrained`

where `TASK_ID` could be ranged in `[0, 1, 2, 3, 4, 5]` corresponding to different network architectures as shown in `train.py`.


## Inference

The model used in the inference stage is set to **ResNet18** as default. To do inference on a audio file, run:

`python infer.py --track_name {AUDIO_PATH}`

where `AUDIO_PATH` should be the file name. For example, if we want to use one of our mixtape `CA7AX Set #3.ogg`, then place the file under the root directory of this project and run `python infer.py --track_name "CA7AX Set #3.ogg"`. The output would be a numpy array whose length is the number of samples of the track, saved as `CA7AX Set #3.npy`, that indicates the sub-genre predictions of the entire track.


## Demo
The survey results are presented at this [link](https://www.wjx.cn/wxloj/datafullscreen.aspx?activity=173412412).
We used Blender 3.0 to create a demo of the inference results, stored in the same google drive link provided above. If you are interested in our demo project, please contact xl3133@nyu.edu for the `.blend` file. The demos are available at [demos of the CA7AX visualizer (v1.0)](https://drive.google.com/drive/u/0/folders/1Vmo9jMYbo0p1agjDoga7Onj1i4WbI4Wk).

## License

This dataset is for research purpose **ONLY**. It is strictly forbidden to use it for ANY commercial use. You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purpose, any portion of the contexts and any portion of derived data. Please contact us through xl3133@nyu.edu if you find you copyright violated by the use of this dataset.


## Acknowledgement

We sincerely appreciate the efforts of musicians who have produced these amazing tracks. We thank members from New York University who have filled the questionnaire of the correspondence between colors and the music sub-genres, for our choices of colors in our demo. Besides, the Blender environment used in the demo is inspired by the splendid work of [Ducky 3D](https://www.youtube.com/channel/UCuNhGhbemBkdflZ1FGJ0lUQ). Also, we thank Wang et al. for assistance on the use of the high performance computing cluster (NYU HPC), and Xia et al. for some valuable advice on our pipeline.
