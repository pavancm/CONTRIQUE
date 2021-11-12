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

### Obtaining Quality Scores
We provide trained regressor models in [models](models) directory which can be used for predicting image quality using features obtained from CONTRIQUE model. For demonstration purposes, some sample images provided in the [sample_images](sample_images) folder.

For blind quality prediction, the following command can be used.
```
python3 demos_score.py --

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
