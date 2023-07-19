
# for RUN_NAME in reward_prediction_seed0_2023_05_26_19_52_05 reward_prediction_seed1_2023_05_26_19_52_26 reward_prediction_seed2_2023_05_26_19_52_19 reward_prediction_seed3_2023_05_26_19_52_39 reward_prediction_seed4_2023_05_26_19_53_31 reward_prediction_seed5_2023_05_26_19_53_31 
# do 
#     for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
#     do
#         CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/CIFAR10/eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE
#     done
# done


# for RUN_NAME in xent+1_seed0_2023_05_26_19_51_24 xent+1_seed1_2023_05_26_19_51_41 xent+1_seed2_2023_05_26_19_51_48 xent+1_seed3_2023_05_26_19_52_09 xent+1_seed4_2023_05_26_19_51_48 xent+1_seed5_2023_05_26_19_52_09
# do 
#     for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
#     do
#         CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/CIFAR10/eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE
#     done
# done


# for SEED in 0 1 2 3 4
# do
#     CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/Amazon/train.py --seed=$SEED
# done 

# for RUN_NAME in xent_seed1_2023_06_21_13_55_05 xent_seed2_2023_06_21_23_47_33 xent_seed3_2023_06_22_09_40_58 xent_seed4_2023_06_22_19_33_20
# do
#     CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/Amazon/eval.py --run-name=$RUN_NAME
# done

# for RUN_NAME in xent_seed0_2023_07_14_17_24_17 xent_seed1_2023_07_14_17_33_57 xent_seed2_2023_07_14_19_32_25 xent_seed3_2023_07_14_19_32_37 xent_seed4_2023_07_14_19_32_49
# do
#     CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/Waterbirds/eval.py --run-name=$RUN_NAME
# done

# for RUN_NAME in reward_prediction_seed0_2023_06_05_16_12_42 reward_prediction_seed1_2023_06_05_21_18_16 reward_prediction_seed2_2023_06_05_21_18_28 reward_prediction_seed3_2023_06_05_21_18_44 reward_prediction_seed4_2023_06_05_21_19_32 xent+1_seed1_2023_07_11_11_46_52 xent+1_seed2_2023_07_11_11_49_46 xent+1_seed3_2023_07_11_11_50_16 xent+1_seed4_2023_07_11_11_50_46 xent+1_seed5_2023_07_11_11_51_15
# do
#     CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/OfficeHome/eval.py --run-name=$RUN_NAME
# done


for RUN_NAME in seed0_2023_06_12_17_43_26 seed1_2023_06_12_18_22_14 seed2_2023_06_12_18_25_32 seed3_2023_06_12_18_35_49 seed4_2023_06_12_18_32_12
do
    CUDA_VISIBLE_DEVICES=1 python ../cautious_extrapolation/UTKFace/eval.py --run-name=$RUN_NAME --corruption-type=impulse_noise
done