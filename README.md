# On Frequency Domain Adversarial Vulnerabilities of Volumetric Medical Image Segmentation (ISBI'25)

> [**On Frequency Domain Adversarial Vulnerabilities of Volumetric Medical Image Segmentation**]()<br><br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ), [Muzammal Naseer](https://scholar.google.com/citations?hl=en&user=tM9xKA8AAAAJ), [Salman Khan](https://scholar.google.com/citations?hl=en&user=M59O9lkAAAAJ), and
[Fahad Shahbaz Khan](https://scholar.google.com/citations?hl=en&user=zvaeYnUAAAAJ)


[![page](https://img.shields.io/badge/Project-Page-F9D371)](https://asif-hanif.github.io/saa/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()




<hr />

| ![main figure](/media/saa_isbi.png)|
|:--| 
| <p align="justify">...</p> |

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
- **Jan 03, 2025** : Accepted in [ISBI 2025]([https://2024.emnlp.org/](https://biomedicalimaging.org/2025/?__hstc=51849206.b421461b571f5471d6d9b6722d06a2b7.1733757041685.1735923468221.1735969456813.7&__hssc=51849206.1.1735969456813&__hsfp=1009270598)) &nbsp;&nbsp; :confetti_ball: :tada:

<!---
</br>
</br>

## Table of Contents
- [Installation](#installation)
- [Model](#model)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Run Experiments](#run-experiments)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

</br>
</br>

<a name="installation"/>

## Installation :gear:
1. Create a conda environment
```shell
conda create --name palm python=3.8
conda activate palm
```
2. Install PyTorch and other dependencies
```shell
git clone https://github.com/asif-hanif/palm
cd palm
pip install -r requirements.txt
```

</br>
<a name="model"/>
    
## Model :white_square_button:
We have shown the efficacy of PALM and other baselines (ZERO-SHOT, COOP, COCOOP) using [PENGI](https://github.com/microsoft/Pengi) model. 

Download the pre-trained PENGI model using the link provided below and place the checkpoint file at path [`pengi/configs`](/pengi/configs) (after clonning the repo). 


| Model | Link | Size |
|:-- |:-- | :-- |
| PENGI | [Download](https://zenodo.org/records/8387083/files/base.pth) | 2.2 GB | 

<br>

PENGI checkpoint can also be downloaded with following command:
```bash
wget https://zenodo.org/records/8387083/files/base.pth
```

</br>

<a name="datasets"/>
    
## Datasets :page_with_curl:

We have performed experiments on 11 audio classification datasets.  Instructions for downloading/processing datasets used by our method have been provided in the [DATASETS.md](DATASETS.md). All of the datasets have been uploaded on HuggingFace Datasets Hub :hugs: for easy access. We have also provided a [Jupyter Notebook](/media/DownloadAudioDatasets.ipynb) to download all datasets in one go. It might take some time to download all datasets, so we recommend running the notebook on a cloud instance or a machine with good internet speed.

| Dataset | Type | Classes | Size | Link |
|:-- |:-- |:--: |--: |:-- |
| [Beijing-Opera](https://compmusic.upf.edu/bo-perc-dataset) | Instrument Classification | 4 | 69 MB | [Instructions](DATASETS.md#beijing-opera) |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Emotion Recognition | 6 | 606 MB | [Instructions](DATASETS.md#crema-d) |
| [ESC50](https://github.com/karolpiczak/ESC-50) | Sound Event Classification | 50 | 881 MB | [Instructions](DATASETS.md#esc50) |
| [ESC50-Actions](https://github.com/karolpiczak/ESC-50) | Sound Event Classification | 10 | 881 MB | [Instructions](DATASETS.md#esc50-actions) |
| [GT-Music-Genre](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | Music Analysis | 10 | 1.3 GB | [Instructions](DATASETS.md#gt-music-genre) |
| [NS-Instruments](https://magenta.tensorflow.org/datasets/nsynth) | Instrument Classification | 10 | 18.5 GB | [Instructions](DATASETS.md#ns-instruments) |
| [RAVDESS](https://zenodo.org/records/1188976#.YFZuJ0j7SL8) | Emotion Recognition | 8 | 1.1 GB | [Instructions](DATASETS.md#ravdess) |
| [SESA](https://zenodo.org/records/3519845) | Surveillance Sound Classification | 4 | 70 MB | [Instructions](DATASETS.md#sesa) |
| [TUT2017](https://zenodo.org/records/400515) | Acoustic Scene Classification | 15 | 12.3 GB | [Instructions](DATASETS.md#tut2017) |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | Sound Event Classification | 10 | 6.8 GB | [Instructions](DATASETS.md#urbansound8k) |
| [VocalSound](https://github.com/YuanGongND/vocalsound) | Vocal Sound Classification | 6 | 8.2 GB | [Instructions](DATASETS.md#vocalsound) |

</br>
</br>

All datasets should be placed in a directory named `Audio-Datasets` and the path of this directory should be specified in the variable `DATASET_ROOT` in the shell [`scripts`](/scripts/). Once all datasets have been downloaded, the directory structure should look like as follows:
```
Audio-Datasets/
    ├── Beijing-Opera/
    ├── CREMA-D/
    ├── ESC50/ 
    ├── ESC50-Actions/
    ├── GT-Music-Genre/
    ├── NS-Instruments/
    ├── RAVDESS/
    ├── SESA/
    ├── TUT2017/
    ├── UrbanSound8K/
    ├── VocalSound/
 ```


</br>

<a name="code-structure"/>

## Code Structure :snowflake:
There are three main folders in this repo: `pengi`, `palm`, `utils`. Code in [`pengi`](/pengi) folder is taken from [PENGI](https://github.com/microsoft/Pengi) repo for model instantiation. Implementation of baselines (`zeroshot`, `coop`, `cocoop`) and our method `palm` is in [`palm`](/palm) folder. Class definitions of audio and text encoder of PENGI model can be found in [`palm/encoders.py`](/palm/encoders.py) file. Training and dataset related code is in [`utils`](/utils) folder.

</br>

<a name="run-experiments"/>

## Run Experiments :zap:

We have performed all experiments on `NVIDIA A100-SXM4-40GB` GPU. Shell scripts to run experiments can be found in [`scripts`](/scripts/) folder. 

```shell
## General Command Structure
bash  <SHELL_SCRIPT>  <METHOD_NAME>
```

Following methods (including `palm`) are supported in this repository:

`zeroshot` `coop` `cocoop` `palm`

Examples to run `palm` method on different audio classifiction datasets have been provided below:

```shell
bash scripts/beijing_opera.sh palm
bash scripts/crema_d.sh palm
bash scripts/esc50_actions.sh palm
bash scripts/esc50.sh palm
bash scripts/gt_music_genre.sh palm
bash scripts/ns_instruments.sh palm
bash scripts/ravdess.sh palm
bash scripts/sesa.sh palm
bash scripts/tut.sh palm
bash scripts/urban_sound.sh palm
bash scripts/vocal_sound.sh palm
```

Results are saved in `json` format in [`logs`](/logs) directory. To process results (take an average across all folds/seeds and print), run the following command (after running all experiments):

```bash
cd logs
bash results.sh
```

<details>
<summary>Sample Output</summary>

![main figure](/media/print_results.png)

</details>

**Note** For multi-fold datasets, we run experiments using cross-validation and then report average results on each seed. 

</br>

<a name="results"/>

## Results :microscope:

<div class="content has-text-justified"><p>
<b>Comparison of PALM with Baselines</b> The accuracy scores of the baselines (<a href=”https://github.com/microsoft/Pengi”>ZERO-SHOT</a>, <a href="https://github.com/KaiyangZhou/CoOp">COOP</a> and <a href="https://github.com/KaiyangZhou/CoOp">COCOOP</a>, and our proposed method PALM) across 11 datasets are presented. For each method (except ZERO SHOT), experiments were performed using three different seeds. The accuracy scores for all seeds are reported, along with the average score. Bold values indicate the best average score in each row. Compared to the baselines, our proposed method achieves favorable results, with an average improvement of 5.5% over COOP and 3.1% over COCOOP. It should be noted that both COOP and COCOOP are computationally expensive, as these approaches require loss gradients to flow through the text encoder. Additionally, COCOOP has a feedback loop from audio features to the input space of the text encoder, making it even more computationally expensive. On the other hand, PALM is relatively less computationally expensive.
</p></div>

![main figure](/media/results.png)

</br>
</br>

<div class="content has-text-justified">
<p align="justify"><b>Comparison of PALM<sup>&dagger;</sup> and PALM</b> Here, <b>PALM<sup>&dagger;</sup></b> refers to setting in which the <i>Learnable Context</i> embeddings have been <b>removed</b> from the feature space of the text encoder. The removal of context embeddings drastically degrades performance, highlighting their importance.</p>
</div>

![main figure](/media/palm_vs_palm_dagger.png)


</br>

<a name="citation"/>

## Citation :star:
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@article{hanif2024palm,
  title={PALM: Few-Shot Prompt Learning for Audio Language Models},
  author={Hanif, Asif and Agro, Maha Tufail and Qazi, Mohammad Areeb and Aldarmaki, Hanan},
  journal={arXiv preprint arXiv:2409.19806},
  year={2024}
}
```

</br>

<a name="contact"/>

## Contact :mailbox:
Should you have any questions, please create an issue on this repository or contact us at **asif.hanif@mbzuai.ac.ae**

</br>

<a name="acknowledgement"/>

## Acknowledgement :pray:
We used [PENGI](https://github.com/microsoft/Pengi) for model instantiation and borrowed a part of code from [COOP/COCOOP](https://github.com/KaiyangZhou/CoOp) to implement baselines. We thank the respective authors for releasing the code.

<hr />

-->


