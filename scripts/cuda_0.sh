
# for RUN_NAME in xent_seed0_2023_05_26_19_49_28 xent_seed1_2023_05_26_19_49_29 xent_seed2_2023_05_26_19_50_57 xent_seed3_2023_05_26_19_50_57 xent_seed4_2023_05_26_19_50_57 xent_seed5_2023_05_26_19_51_22
# do 
#     CUDA_VISIBLE_DEVICES=0 python ../cautious_extrapolation/CIFAR10/calibrate.py --run-name=$RUN_NAME
#     for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
#     do
#         for CORRUPTION_LEVEL in 0 1 2 3 4
#         do
#             CUDA_VISIBLE_DEVICES=0 python ../cautious_extrapolation/CIFAR10/calibrate.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE --corruption-level=$CORRUPTION_LEVEL
#         done
#     done
# done


for RUN_NAME in xent_seed0_2023_05_26_19_49_28 xent_seed1_2023_05_26_19_49_29 xent_seed2_2023_05_26_19_50_57 xent_seed3_2023_05_26_19_50_57 xent_seed4_2023_05_26_19_50_57 xent_seed5_2023_05_26_19_51_22
do 
    for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
    do
        CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/CIFAR10/eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE
    done
done