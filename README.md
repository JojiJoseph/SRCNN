# SRCNN

Implementation of the paper [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)

# How to Run

Create a folder named `datasets` in the root folder of the source.

Download [T91](https://www.kaggle.com/datasets/ll01dm/t91-image-dataset) dataset and store it in datasets folder.

Download [Set5 and Set14](https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset) datasets and store it in datasets folder.

Prepare subimage dataset by

```
python3 make_subimage_dataset.py
```

To train,

```
python3 train.py
```

To see qualitative results,

```
python3 test.py
```
