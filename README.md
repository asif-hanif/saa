# On Frequency Domain Adversarial Vulnerabilities of Volumetric Medical Image Segmentation (ISBI'25)

> [**On Frequency Domain Adversarial Vulnerabilities of Volumetric Medical Image Segmentation**]()<br><br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ), [Muzammal Naseer](https://scholar.google.com/citations?hl=en&user=tM9xKA8AAAAJ), [Salman Khan](https://scholar.google.com/citations?hl=en&user=M59O9lkAAAAJ), and
[Fahad Shahbaz Khan](https://scholar.google.com/citations?hl=en&user=zvaeYnUAAAAJ)


[comment]: [![page](https://img.shields.io/badge/Project-Page-F9D371)](https://asif-hanif.github.io/saa/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10981075)




<hr />

| ![main figure](/media/saa_isbi.png)|
|:--| 
| <p align="justify">**Spectrum Adversarial Attack** $(\mathsf{SA}^2)$ partitions the clean volumetric image into 3D patches, applies 3D-DCT to each patch, and amplifies/attenuates the DCT coefficients using multiplicative spectral noise $(\xi)$. The perturbed spectrum is then converted back to the voxel domain via 3D-IDCT. The loss gradient flow to $\xi$ is shown by the black dashed line.</p> |

</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
In safety-critical domains like healthcare, resilience of deep learning models towards adversarial attacks is crucial. Volumetric medical image segmentation is a fundamental task, providing critical insights for diagnosis. This paper introduces a novel frequency domain adversarial attack targeting 3D medical data, revealing vulnerabilities in segmentation models. By manipulating the frequency spectrum (low, middle, and high bands), we assess its impact on model performance. Unlike pixel-based 2D attacks, our method continuously perturbs 3D samples with minimal information loss, achieving high fooling rates at lower computational costs and superior black-box transferability, while maintaining perceptual quality. 
<br><br>
</i></p>

> <b>TLDR:</b> A novel frequency domain attack on 3D medical segmentation exposes vulnerabilities, achieving high fooling rates, low cost, and superior transferability while preserving perceptual quality.

</br>
</br>

## Updates :rocket:
- **Jan 03, 2025** : Accepted in [ISBI 2025](https://biomedicalimaging.org/2025/) &nbsp;&nbsp; :confetti_ball: :tada:
- **April 10, 2025** : Code released


</br>
</br>

## Table of Contents
- [Installation](#installation)
- [Models](#models)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Experiments](#experiments)
- [Results](#results)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

</br>
</br>

<a name="installation"/>

## Installation :gear:
1. Create a conda environment
```shell
conda create --name saa python=3.8
conda activate saa
```
2. Install PyTorch and other dependencies
```shell
git clone https://github.com/asif-hanif/saa
cd saa
pip install -r requirements.txt
```


<a name="models"/>

## Models :white_square_button:
We have used three volumetric medical image segmentation models: [UNET](), [UNETR]() and [Swin-UNETR]()



<a name="datasets"/>

## Datasets :page_with_curl:
We conducted experiments on two volumetric medical image segmentation datasets: [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). Synapse contains 14 classes (including background) and ACDC contains 4 classes (including background). We follow the same dataset preprocessing as in [nnFormer](https://github.com/282857341/nnFormer). 

The dataset folders for Synapse should be organized as follows: 

```
DATASET_SYNAPSE/
    ├── imagesTr/
        ├── img0001.nii.gz
        ├── img0002.nii.gz
        ├── img0003.nii.gz
        ├── ...  
    ├── labelsTr/
        ├── label0001.nii.gz
        ├── label0002.nii.gz
        ├── label0003.nii.gz
        ├── ...  
    ├── dataset_synapse_18_12.json
 ```

File `dataset_synapse_18_12.json` contains train-val split (created from train files) of Synapse dataset. There are 18 train images and 12 validation images. File `dataset_synapse_18_12.json` can be accessed [here](miscellaneous/dataset_synapse_18_12.json). Place this file in dataset parent folder. Pre-processed Synapse dataset can be downloaded from the following link as well.

| Dataset | Link |
|:-- |:-- |
| BTCV-Synapse (18-12 split) | [Download](https://drive.google.com/file/d/1-Tst3l2kMrC0rlNGDM9CwvRk_2KRFXOo/view?usp=sharing) |

You can use the command `tar -xzf btcv-synapse.tar.gz` to uncompress the file.

</br>

</br>

<a name="code-structure"/>

## Code Structure :snowflake:
The repository is organized as follows:

```
saa/
├── attacks/                    # Adversarial attack implementations
│   ├── saa/                    # Spectrum Adversarial Attack (SAA) implementation
│   │   ├── saa.py              # Main SAA attack class (3D)
│   │   ├── compression.py      # 3D-DCT compression utilities
│   │   ├── decompression.py    # 3D-IDCT decompression utilities
│   │   └── utils.py            # SAA helper functions
│   ├── vafa/                   # VAFA attack implementation (baseline)
│   │   ├── vafa.py             # VAFA attack class
│   │   └── ...
│   ├── pgd.py                  # Projected Gradient Descent attack
│   ├── fgsm.py                 # Fast Gradient Sign Method attack
│   ├── bim.py                  # Basic Iterative Method attack
│   └── gn.py                   # Gaussian Noise attack
├── utils/                      # Utility functions
│   ├── data_utils.py           # Dataset loading and preprocessing
│   ├── get_args.py             # Command-line argument parsing
│   └── utils.py                # General utility functions
├── optimizers/                 # Optimizer implementations
│   └── lr_scheduler.py         # Learning rate schedulers
├── trainer.py                  # Training loop implementation
├── generate_adv_samples.py     # Script to generate adversarial samples
├── inference_on_saved_adv_samples.py  # Script for inference on saved adversarial samples
├── run_normal_or_adv_training.py      # Script for normal or adversarial training
├── miscellaneous/                     # Miscellaneous files
│   └── dataset_synapse_18_12.json     # Dataset split file
└── media/                             # Media files (figures, images)
```

The main attack implementations are located in the `attacks/` directory. The SAA (Spectrum Adversarial Attack) implementation is in `attacks/saa/`, which contains the core frequency-domain attack using 3D-DCT transformations. The `utils/` directory contains data loading, preprocessing, and general utility functions. Training and inference scripts are in the root directory.

</br>




<a name="run-experiments"/>

## Run Experiments :zap:

<a name="launch-saa-attack-on-the-model"/>

## Launch SAA Attack on the Model
```shell
python generate_adv_samples.py --model_name unet-r --feature_size=16 --infer_overlap=0.5 \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--gen_val_adv_mode \
--save_adv_images_dir=<PATH_TO_SAVE_ADV_TEST_IMAGES> \
--attack_name saa --rho 0.2 --steps 20 --block_size 32 32 32 --lambda_dice 0.01 --use_ssim_loss --debugging
```
If adversarial images are not intended to be saved, use `--debugging` argument. If `--use_ssim_loss` is not mentioned, SSIM loss will not be used in the adversarial objective. If adversarial versions of train images are intended to be generated, mention argument `--gen_train_adv_mode` instead of `--gen_val_adv_mode`.

For SAA attack on each 2D slice of volumetric image, use : `--attack_name saa-2d --rho 0.2 --steps 20 --block_size 32 32 --lambda_dice 0.01 --use_ssim_loss`

**Note**: The codebase also supports VAFA attack variants. For VAFA-3D, use: `--attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss`. For VAFA-2D, use: `--attack_name vafa-2d --q_max 20 --steps 20 --block_size 32 32 --use_ssim_loss`

Use following arguments when launching pixel/voxel domain attacks:

[PGD](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.pgd):&nbsp;&nbsp;&nbsp;        `--attack_name pgd --steps 20 --eps 4 --alpha 0.01`

[FGSM](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.fgsm):             `--attack_name fgsm --steps 20 --eps 4 --alpha 0.01`

[BIM](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.bim):&nbsp;&nbsp;&nbsp;        `--attack_name bim --steps 20 --eps 4 --alpha 0.01`

[GN](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.gn):&nbsp;&nbsp;&nbsp;&nbsp;   `--attack_name gn --steps 20 --eps 4 --alpha 0.01 --std 4`

<a name="launch-adversarial-training-of-the-model"/>

## Launch Adversarial Training of the Model
```shell
python run_normal_or_adv_training.py --model_name unet-r --in_channels 1 --out_channel 14 --feature_size=16 --batch_size=3 --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_checkpoint \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--save_model_dir=<PATH_TO_SAVE_ADVERSARIALLY_TRAINED_MODEL> \
--val_every 15 \
--adv_training_mode --freq_reg_mode \
--attack_name saa --rho 0.2 --steps 20 --block_size 32 32 32 --lambda_dice 0.01 --use_ssim_loss 
```

Argument `--adv_training_mode` in conjunction with `--freq_reg_mode` performs adversarial training with dice loss on clean images, adversarial images and frequency regularization term in the objective function. For vanilla adversarial training (i.e. dice loss on adversarial images only), use only `--adv_training_mode`. For normal training of the model, do not mention these two arguments. 


<a name="inference-on-the-model-with-already-saved-adversarial-images"/>

## Inference on the Model with already saved Adversarial Images
If adversarial images have already been saved and one wants to do inference on the model using saved adversarial images, use following command:

```shell
python inference_on_saved_adv_samples.py --model_name unet-r --in_channels 1 --out_channel 14 --feature_size=16 --infer_overlap=0.5 \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--adv_images_dir=<PATH_OF_SAVED_ADVERSARIAL_IMAGES> \ 
--attack_name saa --rho 0.2 --steps 20 --block_size 32 32 32 --lambda_dice 0.01 --use_ssim_loss 
```

Attack related arguments are used to automatically find the sub-folder containing adversarial images. Sub-folder should be present in parent folder path specified by `--adv_images_dir` argument. If `--no_sub_dir_adv_images` is mentioned, sub-folder will not be searched and images are assumed to be present directly in the parent folder path specified by `--adv_images_dir` argument. Structure of dataset folder should be same as specified in [Dataset](#dataset) section.


## Results :microscope:

<a name="results"/>

## Results :microscope:

<div class="content has-text-justified"><p>


![main figure](/media/attack_and_defense_results.png)

</br>
</br>

<a name="citation"/>

## Citation :star:

If you find our work, or this repository useful, please consider giving a star :star: and citation.
```bibtex
@inproceedings{hanif2025frequency,
  title={On Frequency Domain Adversarial Vulnerabilities of Volumetric Medical Image Segmentation},
  author={Hanif, Asif and Naseer, Muzammal and Khan, Salman and Khan, Fahad Shahbaz},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={01--05},
  year={2025},
  organization={IEEE}
}
```

<hr />

<a name="contact"/>

## Contact
Should you have any question, please create an issue on this repository or contact at **asif.hanif@mbzuai.ac.ae**

<hr />

<a name="acknowledgement"/>

## Acknowledgement
We thank the authors of [UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV), [nnFormer](https://github.com/282857341/nnFormer), and [MONAI](https://github.com/Project-MONAI/MONAI) for releasing their code and models.

<hr />


