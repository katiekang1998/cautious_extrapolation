
for RUN_NAME in reward_prediction_seed0_2023_05_26_19_52_05 reward_prediction_seed1_2023_05_26_19_52_26 reward_prediction_seed2_2023_05_26_19_52_19 reward_prediction_seed3_2023_05_26_19_52_39 reward_prediction_seed4_2023_05_26_19_53_31 reward_prediction_seed5_2023_05_26_19_53_31 
do 
    for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
    do
        CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/CIFAR10/eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE
    done
done


for RUN_NAME in xent+1_seed0_2023_05_26_19_51_24 xent+1_seed1_2023_05_26_19_51_41 xent+1_seed2_2023_05_26_19_51_48 xent+1_seed3_2023_05_26_19_52_09 xent+1_seed4_2023_05_26_19_51_48 xent+1_seed5_2023_05_26_19_52_09
do 
    for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
    do
        CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/CIFAR10/eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE
    done
done