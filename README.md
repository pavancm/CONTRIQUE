# Image Quality Assessment using Contrastive Learning

**Pavan C. Madhusudana**, Neil Birkbeck, Yilin Wang, Balu Adsumilli and Alan C. Bovik

This is the official repository of the paper [Image Quality Assessment using Contrastive Learning](https://arxiv.org/abs/2110.13266)

## Usage
The code has been tested on Linux systems with python 3.6. Please refer to [requirements.txt](requirements.txt) for installing dependent packages.

### Running CONTRIQUE
In order to obtain quality score using CONTRIQUE model, checkpoint needs to be downloaded. The following command can be used to download the checkpoint.
```
wget -L https://utexas.box.com/shared/static/rhpa8nkcfzpvdguo97n2d5dbn4qb03z8.tar -O models/CONTRIQUE_checkpoint25.tar -q --show-progress
```
Alternatively, the checkpoint can also be downloaded using this [link](https://utexas.box.com/s/rhpa8nkcfzpvdguo97n2d5dbn4qb03z8).

Google drive link for the checkpoint [link](https://drive.google.com/file/d/1pmaomNVFhDgPSREgHBzZSu-SuGzNJyEt/view?usp=drive_web).

### Obtaining Quality Scores
We provide trained regressor models in [models](models) directory which can be used for predicting image quality using features obtained from CONTRIQUE model. For demonstration purposes, some sample images provided in the [sample_images](sample_images) folder.

For blind quality prediction, the following commands can be used.
```
python3 demo_score.py --im_path sample_images/60.bmp --model_path models/CONTRIQUE_checkpoint25.tar --linear_regressor_path models/CLIVE.save
python3 demo_score.py --im_path sample_images/img66.bmp --model_path models/CONTRIQUE_checkpoint25.tar --linear_regressor_path models/LIVE.save
```

For Full-reference quality assessment, the folllowing command can be employed.
```
python3 demos_score_FR.py --ref_path sample_images/churchandcapitol.bmp --dist_path sample_images/img66.bmp --model_path models/CONTRIQUE_checkpoint25.tar --linear_regressor_path models/CSIQ_FR.save
```

### Obtaining CONTRIQUE Features
For calculating CONTRIQUE features, the following commands can be used. The features are saved in '.npy' format.
```
python3 demo_feat.py --im_path sample_images/60.bmp --model_path models/CONTRIQUE_checkpoint25.tar --feature_save_path features.npy
python3 demo_feat.py --im_path sample_images/img66.bmp --model_path models/CONTRIQUE_checkpoint25.tar --feature_save_path features.npy
```

## Training CONTRIQUE
### Download Training Data
Create a directory ```mkdir training_data``` to store images used for training CONTRIQUE.
1. KADIS-700k : Download [KADIS-700k](http://database.mmsp-kn.de/kadid-10k-database.html) dataset and execute the supllied codes to generate synthetically distorted images. Store this data in the ```training_data/kadis700k``` directory.
2. AVA : Download [AVA](https://github.com/mtobeiyf/ava_downloader) dataset and store in the ```training_data/UGC_images/AVA_Dataset``` directory.
3. COCO : [COCO](https://cocodataset.org/#download) dataset contains 330k images spread across multiple competitions. We used 4 folders ```training_data/UGC_images/test2015, training_data/UGC_images/train2017, training_data/UGC_images/val2017, training_data/UGC_images/unlabeled2017``` for training.
4. CERTH-Blur : [Blur](https://mklab.iti.gr/results/certh-image-blur-dataset/) dataset images are stored in the ```training_data/UGC_images/blur_image``` directory.
5. VOC : [VOC](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/) images are stored in the ```training_data/UGC_images/VOC2012``` directory.

### Training Model
Download csv files containing path to images and corresponding distortion classes.
```
wget -L https://utexas.box.com/shared/static/124n9sfb27chgt59o8mpxl7tomgvn2lo.csv -O csv_files/file_names_ugc.csv -q --show-progress
wget -L https://utexas.box.com/shared/static/jh5cmu63347auyza37773as5o9zxctby.csv -O csv_files/file_names_syn.csv -q --show-progress
```
The above files can also be downloaded manually using these links [link1](https://utexas.box.com/s/jh5cmu63347auyza37773as5o9zxctby), [link2](https://utexas.box.com/s/124n9sfb27chgt59o8mpxl7tomgvn2lo)
Google drive links [link1](https://drive.google.com/file/d/1uKcTJ5ioVpOkQ-s7mOOhZNke7XCi9i4Q/view?usp=drive_web), [link2](https://drive.google.com/file/d/11bfkLaAFT7CN_z6fQbC8Zg8yiusIBnE6/view?usp=drive_web)



For training with a single GPU the following command can be used
```
python3 train.py --batch_size 256 --lr 0.6 --epochs 25
```

Training with multiple GPUs using Distributed training (Recommended)

Run the following commands on different terminals concurrently
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --nodes 4 --nr 0 --batch_size 64 --lr 0.6 --epochs 25
CUDA_VISIBLE_DEVICES=1 python3 train.py --nodes 4 --nr 1 --batch_size 64 --lr 0.6 --epochs 25
CUDA_VISIBLE_DEVICES=2 python3 train.py --nodes 4 --nr 2 --batch_size 64 --lr 0.6 --epochs 25
CUDA_VISIBLE_DEVICES=3 python3 train.py --nodes 4 --nr 3 --batch_size 64 --lr 0.6 --epochs 25
```
Note that in distributed training, ```batch_size``` value will be the number of images to be loaded on each GPU. During CONTRIQUE training equal number of images will be loaded from both synthetic and authentic distortions. Thus in the above example code, 128 images will be loaded on each GPU.

### Training Linear Regressor
After CONTRIQUE model training is complete, a linear regressor is trained using CONTRIQUE features and corresponding ground truth quality scores using the following command.

```
python3 train_regressor.py --feat_path feat.npy --ground_truth_path scores.npy --alpha 0.1
```

## Contact
Please contact Pavan (pavan.madhusudana@gmail.com) if you have any questions, suggestions or corrections to the above implementation.

## Citation
```
@article{madhusudana2021st,
  title={Image Quality Assessment using Contrastive Learning},
  author={Madhusudana, Pavan C and Birkbeck, Neil and Wang, Yilin and Adsumilli, Balu and Bovik, Alan C},
  journal={arXiv:2110.13266},
  year={2021}
}
```
