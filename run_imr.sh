# 10 tasks
#python main.py imr_RainbowPrompt --model vit_base_patch16_224 --output_dir ./output_IMR_10task --trial IMR_10task  --device cuda:3 --epochs 50  \
#                                         --batch-size 32  --length 20  --top_k 1 --size 10  --D1 56 --D2 96  --warm_up 5 \
#                                         --use_linear True --relation_type attention --balancing 0.01 \
#                                         --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 --self_attn_idx 0 1 2 3 4 5 --num_tasks 10 --seed 23846

# 20 tasks
python main.py imr_RainbowPrompt --model vit_base_patch16_224 --output_dir ./output_IMR_20task --trial IMR_20task  --device cuda:3 --epochs 100  \
                                         --batch-size 32  --length 20  --top_k 1 --size 20  --D1 28 --D2 56  --warm_up 5 \
                                         --use_linear True --relation_type attention --balancing 0.01 \
                                         --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 --self_attn_idx 0 1 2 3 4 5 --num_tasks 20

