# 10 tasks
#python main.py cub200_RainbowPrompt --model vit_base_patch16_224 --output_dir ./output_CUBS_10task --trial CUBS_10task  --device cuda:1 --epochs 20  \
#                                         --batch-size 32  --length 20  --top_k 1 --size 10  --D1 56 --D2 96  --warm_up 5 \
#                                         --use_linear True --relation_type attention --balancing 0.01  \
#                                         --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 --self_attn_idx 0 1 2 3 4 5 --num_tasks 10 --seed 99790

# 20 tasks
python main.py cub200_RainbowPrompt --model vit_base_patch16_224 --output_dir ./output_CUBS_20task --trial CUBS_20task  --device cuda:1 --epochs 20  \
                                         --batch-size 32  --length 20  --top_k 1 --size 20  --D1 28 --D2 56  --warm_up 5 \
                                         --use_linear True --relation_type attention --balancing 0.01  \
                                         --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 --self_attn_idx 0 1 2 3 4 5 --num_tasks 20 