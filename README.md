# cnn_depth_tensorflow
cnn_depth_tensorflow is an implementation of depth estimation using tensorflow.

Original paper is "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network".
https://arxiv.org/abs/1406.2283

![network](images/network.png)

# requierments
- TensorFlow 0.10+ or TensorFlow GPU
```
pip install tensorflow==1.0.0 
# or
pip install tensorflow-gpu==1.15
```

- Numpy
```
pip install numpy
```
- Wget
```
pip install wget
```
- h5py
```
pip install h5py
```
- PIL
```
pip install Pillow
```

# How to train
- Download training data. Please see readme.md in data directory.
- Convert mat to png images.
```
python prepare_data.py
```

- Lets's train.
```
python task.py
```

- You can see predicting images through training in data directory.

# Eval
```
python eval.py
```

# example
- input  
<img src="images/input.png" width="200">
- output  
<img src="images/output.png" width="200">

---

Copyright (c) 2016 Masahiro Imai
Released under the MIT license
