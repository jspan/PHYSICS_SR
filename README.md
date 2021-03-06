# PHYSICS_SR

This repository is an official PyTorch implementation of the paper "Image Formation Model Guided Deep Image Super-Resolution" from AAAI 2020.  
The code is built on EDSR (PyTorch) and tested on Ubuntu 16.04 environment (Python3.6, PyTorch_0.4.1, CUDA8.0, cuDNN5.1) with Tesla V100/1080Ti GPUs.


## Dependencies
* ubuntu16.04
* Python 3.6(Recommend to use Anaconda)
* PyTorch0.4.1
* numpy
* skimage
* imageio
* matplotlib
* tqdm
* cv2 

## Get started

#### Training dataset:
We use the DIV2K dataset to train our models. You can download it from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

#### Benechmarks:
You can evaluate our models with widely-used benchmark datasets:

[Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html),

[Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests),

[B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),

[Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).

#### Models
All the models(X2, X3, X4) can be downloaded from [GoogleDrive](https://drive.google.com/open?id=1ns0zFBZgOCFxafBALRq7_wO-jrUZqNfH).

## Quicktest with benchmark
You can test our super-resolution algorithm with benchmarks. Please organize the testset in  ``testset`` folder like this:  
```
|--testset  
    |--Set5  
        |--LR
            |--X2
                |--babyx2.png  
                     ：   
                     ： 
            |--X3
            |--X4
        |--HR
            |--baby.png  
                 ：   
                 ： 
```
    
Then, run the following commands:
```bash
cd code
python main.py --dir_data ../testset --data_test Set5 --model physics_sr --pre_train ../models/X2/model_best.pt --scale 2 --save physics_sr_x2 --save_results --test_only
```
And generated results can be found in ``./experiment/physics_sr_x2/results/``
  * To test all scales, you can modify the options(pre_train, scale) of the command above.  
  * To test other benchmarks, you can modify the option(data_test) of the command above.   
  * To change the save root, you can modify the option(save) of the command above.  
  

## How to train
If you have downloaded the trainset, please make sure that the trainset has been organized as follows:
```
|--DIV2K
    |--train  
        |--DIV2K_train_LR_bicubic
            |--X2
                |--0001x2.png  
                     ：   
                     ： 
            |--X3
            |--X4
        |--DIV2K_train_HR
            |--0001.png  
                 ：   
                 ： 
```
The command for training is as follow:
```
cd code
python main.py --dir_data <your root> --data_test DIV2K --model physics_sr --scale 2 --save physics_sr_x2 --save_results
```
The trained model can be found in ``./experiment/physics_sr_x2/model``
  * To train all scales, you can modify the option(scale) of the command above.  
  * To test other benchmarks, you can modify the option(data_test) of the command above.   
  * To change the save root, you can modify the option(save) of the command above.  


## Citation
If our work is useful in your research or publication, please cite our work:
```

@inproceedings{pan2020physics_sr,
    title={Image Formation Model Guided Deep Image Super-Resolution},
    author={Jinshan Pan, Yang Liu, Deqing Sun, Jimmy Ren, Ming-Ming Cheng, Jian Yang, and Jinhui Tang},
    booktitle={AAAI},
    year={2020}
}
```

## Acknowledgements
This code is built on EDSR(PyTorch). We thank the authors for sharing their codes of EDSR.
