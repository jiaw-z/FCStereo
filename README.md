# Revisiting Domain Generalized Stereo Matching Networks from a Feature Consistency Perspective (CVPR2022)

This is the implementation of the paper **Revisiting Domain Generalized Stereo Matching Networks from a Feature Consistency Perspective**, CVPR 2022, Jiawei Zhang, Xiang Wang, Xiao Bai, Chen Wang, Lei Huang, Yimin Chen, Lin Gu, Jun Zhou, Tatsuya Harada, Edwin R. Hancock 

paper: [[arxiv](https://arxiv.org/pdf/2203.10887.pdf)]

The code is based on [DenseMatchingBenchmark](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark).

## Use of DenseMatchingBenchmark
### Installation
Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.
### Get Started
Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of DenseMatchingBenchmark.

## Abstract
Despite recent stereo matching networks achieving impressive performance given sufficient training data, they suffer from domain shifts and generalize poorly to unseen domains. We argue that maintaining feature consistency between matching pixels is a vital factor for promoting the generalization capability of stereo matching networks, which has not been adequately considered. Here we address this issue by proposing a simple pixel-wise contrastive learning across the viewpoints. The stereo contrastive feature loss function explicitly constrains the consistency between learned features of matching pixel pairs which are observations of the same 3D points. A stereo selective whitening loss is further introduced to better preserve the stereo feature consistency across domains, which decorrelates stereo features from stereo viewpoint-specific style information. Counter-intuitively, the generalization of feature consistency between two viewpoints in the same scene translates to the generalization of stereo matching performance to unseen domains. Our method is generic in nature as it can be easily embedded into existing stereo networks and does not require access to the samples in the target domain. When trained on synthetic data and generalized to four real-world testing sets, our method achieves superior performance over several state-of-the-art networks.


## Method overview
![image](https://user-images.githubusercontent.com/66359549/159516301-05ad393d-c710-4037-8826-ce68778f9330.png)

## Method implementation
### config
[./configs/FCStereo/](./configs/FCStereo/)
### Stereo Contrastive Feature (SCF) loss
[./dmb/modeling/stereo/models/fc_stereo_base.py](./dmb/modeling/stereo/models/fc_stereo_base.py)

[./dmb/modeling/stereo/losses/contrastive_loss.py](./dmb/modeling/stereo/losses/contrastive_loss.py)

### Stereo Selective Whitening (SSW) loss
[./dmb/modeling/stereo/losses/ssw_loss.py](./dmb/modeling/stereo/losses/ssw_loss.py)

[./dmb/modeling/stereo/layers/instance_whitening.py](./dmb/modeling/stereo/layers/instance_whitening.py)


## Train from scratch
You can use the [bash file](./tools/dist_train.sh) to train a model.
### FC-PSMNet
You can train PSMNet with our feature consistency constriants with the [config file](./configs/FCStereo/fc_psm_sceneflow.py).

## Citation
If you find this project helpful in your research, welcome to cite the paper:
```
    @article{zhang2022revisiting,
      title={Revisiting Domain Generalized Stereo Matching Networks from a Feature Consistency Perspective},
      author={Zhang, Jiawei and Wang, Xiang and Bai, Xiao and Wang, Chen and Huang, Lei and Chen, Yimin and Gu, Lin and Zhou, Jun and Harada, Tatsuya and Hancock, Edwin R},
      journal={arXiv preprint arXiv:2203.10887},
      year={2022}
    }
```

If you use the DenseMatchingBenchmark toolbox or benchmark in your research, please also consider cite:
```
    @article{zhang2020adaptive,
      title={Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching},
      author={Zhang, Youmin and Chen, Yimin and Bai, Xiao and Yu, Suihanjin and Yu, Kun and Li, Zhiwei and Yang, Kuiyuan},
      journal={AAAI},
      year={2020}
    }
```

## License
This Repo is released under MIT [License](LICENSE).

