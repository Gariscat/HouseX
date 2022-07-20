# HouseX

HouseX is a fine-grained house music dataset, including 160 tracks, that provides 4 sub-genre labels and around 17480 converted mel-spectrograms of slices of these tracks. Besides, this repository includes several baseline models for classification on these mel-spectrograms. Additionally, the start and the end of the drop of each track is annotated, as provided by the json files.


## Dataset
The mel-spectrograms are available at 
[HouseX processed data (Google Drive link)](https://drive.google.com/drive/u/1/folders/1HHi_WadYdea791zOq0Ib07AAPsR__yH-)


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

`python train.py --id {TASK_ID}`

where `TASK_ID` could be ranged in `[0, 1, 2, 3, 4, 5]` corresponding to different network architectures as shown in `train.py`.


## Inference

The format of the input audio file is set to `.ogg` as default. To do inference on a audio file, run:

`python infer.py --track_name {AUDIO_PATH}`

where `AUDIO_PATH` should be the file name **without suffix**. For example, if we want to use one of our mixtape `CA7AX Set #3.ogg`, then place the file under the root directory of this project and run `python infer.py --track_name "CA7AX Set #3"`.


## License

This dataset are for research purpose **ONLY**. It is strictly forbidden to use them for ANY commercial use. You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purpose, any portion of the contexts and any portion of derived data. Please contact us through xl3133@nyu.edu if you find you copyright violated by the use of this dataset.


