# EDSR baseline model (x2)
#python main.py --model EDSR --scale 2 --save EDSR_baseline_x2 --reset

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --save EDSR_baseline_x3 --reset --pre_train ../experiment/model/EDSR_baseline_x2.pt

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save EDSR_baseline_x4 --reset --pre_train ../experiment/model/EDSR_baseline_x2.pt

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save EDSR_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save EDSR_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/EDSR_x2/model/model_best.pt

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save EDSR_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/EDSR_x2/model/model_best.pt

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test Set14 --scale 4 --pre_train ../experiment/EDSR_x4/model/model_best.pt --test_only --self_ensemble
#python main.py --data_test B100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test Urban100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test DIV2K --ext img --n_val 100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble

#python main.py --data_test Set5 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test Set14 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test B100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test Urban100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test DIV2K --ext img --n_val 100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble

# Test your own images
python main.py --data_test Demo --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --save_results

# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train ../experiment/model/MDSR_baseline_jpeg.pt --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save EDSR_GAN --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train ../experiment/model/EDSR_baseline_x4.pt

# my test configuration
srun -p Pixel --gres=gpu:1 --job-name=sr_train --kill-on-bad-exit=1 python main.py --data_test Set5 --model edsr_coarse2fine --pre_train ../experiment/EDSR_x2/model/model_best.pt --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --save EDSR_x2 --save_results --test_only

srun -p Pixel --gres=gpu:1 --job-name=sr_train --kill-on-bad-exit=1 python main.py --data_test Set14 --model edsr_coarse2fine --pre_train ../experiment/EDSR_x2/model/model_best.pt --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --save EDSR_x2 --save_results --test_only

srun -p Pixel --gres=gpu:1 --job-name=sr_train --kill-on-bad-exit=1 python main.py --data_test B100 --model edsr_coarse2fine --pre_train ../experiment/EDSR_x2/model/model_best.pt --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --save EDSR_x2 --save_results --test_only

srun -p Pixel --gres=gpu:1 --job-name=sr_train --kill-on-bad-exit=1 python main.py --data_test Urban100 --model edsr_coarse2fine --pre_train ../experiment/EDSR_x3/model/model_best.pt --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --save EDSR_x3 --save_results --test_only

srun -p Pixel --gres=gpu:4 --job-name=sr_train --kill-on-bad-exit=1 python main.py --data_test DIV2K --model edsr_coarse2fine --pre_train ../experiment/EDSR_x4/model/model_best.pt --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --save EDSR_x4 --n_GPUs 4 --save_results --test_only --chop
