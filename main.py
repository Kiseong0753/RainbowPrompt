
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import models
import utils
from loguru import logger

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_seed(seed):
    logger.info(f"Seed: {seed}")

def initialize_experiment(seed=None):
    if seed == 0:
        seed = random.randint(1, 100000)
    set_seed(seed)
    save_seed(seed)

def main(args):
    
    log_dir = os.path.join('./logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger.add(log_dir + '/' + args.trial +"_{time}.log")
    logger.add(sys.stdout, colorize=True, format="{message}")
    
    args_str = ', '.join(f'{k}={v}' for k, v in vars(args).items())
    logger.info(f'Parsed arguments: {args_str}')
    
    initialize_experiment(seed=args.seed)
    
    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)
    logger.info(f'Class Mask: {class_mask}')

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        n_tasks=args.num_tasks, 
        D1=args.D1,
        relation_type=args.relation_type,
        use_linear=args.use_linear, 
        warm_up=args.warm_up,
        KI_iter=args.KI_iter,
        self_attn_idx = args.self_attn_idx,
        D2=args.D2
    )
    original_model.to(device)
    model.to(device) 


    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
    

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
            
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of params')
    logger.info(n_parameters)
    
     #Exclude specific layers from parameter count
    excluded_layers = {'key_matcher', 'value_matcher', 'query_matcher', 'dense', 'fc1', 'fc2'}
    
    # Count only parameters not in the excluded layers
    n_parameters = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and not any(excluded_layer in name for excluded_layer in excluded_layers)
    )
    
    logger.info('Number of params (excluding specific layers):')
    logger.info(n_parameters)
    
    
    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RainbowPrompt training and evaluation configs')
    
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'CIFAR100_10Task_RainbowPrompt':
        from configs.cifar100_10task_RainbowPrompt import get_args_parser
        config_parser = subparser.add_parser('CIFAR100_10Task_RainbowPrompt', help='10-Split-CIFAR100 RainbowPrompt configs')
    elif config == 'CIFAR100_20Task_RainbowPrompt':
        from configs.cifar100_20task_RainbowPrompt import get_args_parser
        config_parser = subparser.add_parser('CIFAR100_20Task_RainbowPrompt', help='20-Split-CIFAR100 RainbowPrompt configs')
    elif config == 'imr_RainbowPrompt':
        from configs.imr_RainbowPrompt import get_args_parser
        config_parser = subparser.add_parser('imr_RainbowPrompt', help='Split-ImageNet-R RainbowPrompt configs')
    elif config == 'cub200_RainbowPrompt':
        from configs.cub200_RainbowPrompt import get_args_parser
        config_parser = subparser.add_parser('cub200_RainbowPrompt', help='Split-CUB200 RainbowPrompt configs')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
    
    sys.exit(0)
    








 