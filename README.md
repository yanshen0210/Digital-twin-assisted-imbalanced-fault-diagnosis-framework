# Digital twin-assisted imbalanced fault diagnosis framework using subdomain adaptive mechanism and margin-aware regularization
* Core codes for the paper ["Digital twin-assisted imbalanced fault diagnosis framework using subdomain adaptive mechanism and margin-aware regularization"](https://www.sciencedirect.com/science/article/pii/S0951832023004362)
* Created by Shen Yan, Xiang Zhong, Haidong Shao, Yuhang Ming, Chao Liu, Bin Liu.
* Journal: Reliability Engineering and System Safety

![Framework](https://github.com/yanshen0210/Digital-twin-assisted-imbalanced-fault-diagnosis-framework/blob/main/framework.jpg)
## Our operating environment
* Python 3.8
* pytorch  1.10.1
* and other necessary libs

## Guide 
* This repository provides a concise framework for imbalanced fault diagnosis. 
* It includes the pre-processing for the data and the model proposed in the paper. 
* We have also integrated 8 baseline methods including 4 data-level and 4 algorithm-level methods for comparison.
* `train_test.py` is the train&test process of our proposed method; `train_test_base.py` is the train&test process of 8 baseline methods.
* You need to load the data in following Datasets link at first, and put them in the `data` folder. Then run in `args_diagnosis.py`
* You can also choose the modules or adjust the parameters of the model to suit your needs.

## Datasets
* ~~[ADAMS gearbox](https://drive.google.com/file/d/1dlaCZeV-5ACsC1EAuImljdtS53OprZ3V/view?usp=drive_link)~~
(We are very sorry that the simulation data cannot open for the time being because the sharing right is not obtained)
* [SEU gearbox](https://drive.google.com/file/d/1ZfKWYK-xRl3Oy7zMuzlmkSy9G4mQh0y1/view?usp=drive_link)
* [XJTU gearbox](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)

## Run the code
### The proposed method
~~* `args_diagnosis.py` --transfer_task ADAMS_SEU or ADAMS_XJTU; --transfer_loss SAM+MAR~~
* (You can import your own dataset and debug the algorithm)
### data-level methods
* `args_diagnosis.py` --transfer_task SEU or XJTU; --SMOTETomek True; --gan False; --gen_data False
*  `args_diagnosis.py` --transfer_task SEU or XJTU; --SMOTETomek False; --gan True; --gen_data True; --gan_model ACGAN or VAE_GAN or WGAN_GP
### algorithm-level methods
`args_diagnosis.py` --transfer_task SEU or XJTU; --SMOTETomek False; --gan False; --gen_data False; --cost_loss True; --loss WL or FL or DWBL or CBL

## Pakages
* `data` needs loading the Datasets in above links
* `datasets` contians the pre-processing process for the data
* `gans` contians three gan models as baselines
* `loss` contians four types of loss way
* `models` contians the ResNet18 network as the feature extractor
* `utils` contians two types of train&test processes

## Citation
If our work is useful to you, please cite the following paper, it is the greatest encouragement to our open source work, thank you very much!
```
@paper{
  title = {Digital twin-assisted imbalanced fault diagnosis framework using subdomain adaptive mechanism and margin-aware regularization},
  author = {Shen Yan, Xiang Zhong, Haidong Shao, Yuhang Ming, Chao Liu},
  journal = {Reliability Engineering and System Safety},
  volume = {239},
  pages = {109522},
  year = {2023},
  doi = {https://doi.org/10.1016/j.ress.2023.109522},
  url = {https://www.sciencedirect.com/science/article/pii/S0951832023004362},
}
```

## Contact
- yanshen0210@gmail.com
