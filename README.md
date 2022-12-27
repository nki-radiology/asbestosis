# Artificial Intelligence-based diagnosis of asbestosis: analysis of a database with applicants for asbestosis state-aid :computer: :mag_right: 


Managed by Kevin Groot Lipman (k.groot.lipman@nki.nl, k.b.w.grootlipman@gmail.com)

Noninvasive diagnosis of asbestosis for financial compensation suffers from interobserver variability. We developed/integrated an AI-system of multiple components (lung segmentation, anomaly heatmap, classifier) to reproduce the verdict of a panel of medical experts. 

Our paper is published at European Radiology at https://link.springer.com/article/10.1007/s00330-022-09304-2

## 1. Setting up the environment :deciduous_tree:
Create and activate a conda environment with Python
 ```
conda create -n asbestosis python=3.7.9
conda activate asbestosis
 ```
Install the requirements
 ```
conda install tensorflow-gpu==1.15
conda install keras==2.3.1
conda install pandas
conda install matplotlib==3.2.2
conda install scikit-learn=0.24.1
conda install scipy
pip install pynrrd
 ```
Follow instructions at https://github.com/keras-team/keras-contrib to install keras_contrib

## 2. How to run the code 🚀 

### 2.1 VAE.
- Collect a healthy (CT) dataset.
- Adjust the folders to your own paths for this dataset in ``data_generator_vae.py`` and ``train_vae.py``. 
- Run ``python train_vae.py``
- After training, run ``python test_vae.py``

### 2.2 Lung segmentation.
- We adopted the model of Rodney et al. -> https://github.com/lalonderodney/SegCaps

### 2.3 Resnet training.
- Adjust the folders of your target (disease) dataset to your own paths in ``data_generator.py`` and ``train.py``. 
- Run ``python train.py``
- After training, run ``python test.py``

:exclamation: You dont have to have anomaly heatmaps (VAE) or lung segmentations to run it. Just set 'n_channels' in ``train.py`` to 1 and load from your target dataset folder.

## 3. Troubleshooting 🔨 
If you encounter a h5py error:
pip install 'h5py==2.10.0' --force-reinstall

## 4. Contribution :muscle:

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. If you want to add your analysis, or have a suggestion that would make this better, please fork the repo.
