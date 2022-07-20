# HouseX

HouseX is a fine-grained house music dataset, including 160 tracks, that provides 4 sub-genre labels and around 17480 converted mel-spectrograms of slices of these tracks. Besides, this repository includes several baseline models for classification on these mel-spectrograms. Additionally, the start and the end of the drop of each track is annotated, as provided by the json files.


## Dataset
The mel-spectrograms are available at 
[HouseX processed data (Google Drive Link)](https://drive.google.com/drive/u/1/folders/1HHi_WadYdea791zOq0Ib07AAPsR__yH-)


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

Since the computing tasks are done on the NYU HPC cluster, we need to manually load the state dictionary of the pretrained torchvision models. If you want to replicate the experiments, simply delete the line that load state dictionary in `train.py` and set `pretrained = True`. For example, if you want to use pretrained ResNet18, set `backbone = models.resnet18(pretrained=True)`.


## License

This dataset are for research purpose ONLY. It is strictly forbidden to use them for ANY commercial use. You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purpose, any portion of the contexts and any portion of derived data. Please contact us through xl3133@nyu.edu if you find you copyright violated by the use of this dataset.


