
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


# for RUN_NAME in xent_seed0_2023_05_26_19_49_28 xent_seed1_2023_05_26_19_49_29 xent_seed2_2023_05_26_19_50_57 xent_seed3_2023_05_26_19_50_57 xent_seed4_2023_05_26_19_50_57 xent_seed5_2023_05_26_19_51_22
# do 
#     for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
#     do
#         CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/CIFAR10/eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE
#     done
# done

# python ../cautious_extrapolation/ImageNet/train.py --seed=3 --train-type=xent --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:8887' --dist-backend 'nccl' --world-size 1 --rank 0 
# python ../cautious_extrapolation/ImageNet/train.py --seed=4 --train-type=reward_prediction --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:8887' --dist-backend 'nccl' --world-size 1 --rank 0 
# python ../cautious_extrapolation/ImageNet/train.py --seed=4 --train-type=xent --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:8887' --dist-backend 'nccl' --world-size 1 --rank 0 



for RUN_NAME in  reward_prediction_seed1_2023_06_21_09_58_58  reward_prediction_seed2_2023_06_21_21_00_20 reward_prediction_seed3_2023_06_22_07_56_32 reward_prediction_seed4_2023_06_22_23_23_39 xent_seed0_2023_06_21_04_29_07 xent_seed1_2023_06_21_15_30_38 xent_seed2_2023_06_22_02_28_39 xent_seed3_2023_06_22_17_49_30 xent_seed4_2023_06_23_04_54_45
do
    CUDA_VISIBLE_DEVICES=7 python ../cautious_extrapolation/ImageNet/eval.py --run-name=$RUN_NAME
done