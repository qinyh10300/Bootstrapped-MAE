# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch

import copy
from util.ema import EMA
from methods import *

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--name', required=True,
                        help='Name of the checkpoint')
    
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
                        
    parser.add_argument('--optim', default='AdamW', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--ckpt_name', default='checkpoint',
                        help='name of checkpoint')
    parser.add_argument('--current_datetime', required=True,
                        help='current datetime of running the code')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # bootstrapping parameters
    parser.add_argument('--is_bootstrapping', action='store_true')
    parser.set_defaults(is_bootstrapping=False)
    parser.add_argument('--bootstrap_steps', default=5, type=int)
    parser.add_argument('--bootstrap_method', default='Last_layer', type=str)
    parser.add_argument('--feature_layers', default=[1, 6, 12], type=int, nargs='+',
                        help='List of feature layers (e.g., 1 6 12)')
    parser.add_argument('--weights', default=[1, 1, 1], type=float, nargs='+',
                        help='List of weights (e.g., 1 6 12)')

    # ema parameters
    parser.add_argument('--use_ema', action='store_true')
    parser.set_defaults(use_ema=False)
    parser.add_argument('--ema_decay', default=0.99, type=float)
    # parser.add_argument('--ema_lr_decay', default=0.1, type=float)

    # checkpoint saving parameters
    parser.add_argument('--save_frequency', default=20, type=int)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    # print(args.is_bootstrapping)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean/std
            transforms.Normalize(mean=[0.4919, 0.4827, 0.4472], std=[0.2022, 0.1994, 0.2010])])   # CIFAR-10 mean/std
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # TODO: define the model
    if args.is_bootstrapping:
        model = models_mae.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            is_bootstrapping=args.is_bootstrapping,
            bootstrap_method=args.bootstrap_method,
            feature_layers=args.feature_layers,
            )
        
        method_class = None
        seq_len = 64 + 1   # 多一个cls_token
        seq_len += 1       # 使用deit，多一个distill_token
        if args.bootstrap_method == 'Fixed_layer_fusion':
            assert len(args.feature_layers) == len(args.weights), "Length of feature layers and weights must be equal."
            method_class = FixedLayerFusion(args.weights)
        elif args.bootstrap_method == 'Adaptive_layer_fusion':
            method_class = AdaptiveLayerFusion(len(args.feature_layers))
        elif args.bootstrap_method == 'Cross_layer_fusion':
            method_class = CrossLayerFusion(seq_len, len(args.feature_layers)).to(device)
        elif args.bootstrap_method == 'Gated_fusion_dynamic':
            method_class = GatedFusionDynamic(seq_len, len(args.feature_layers)).to(device)
        elif args.bootstrap_method == 'Cross_layer_self_attention':
            method_class = CrossLayerSelfAttention(seq_len, len(args.feature_layers), embed_dim=192).to(device)
        elif args.bootstrap_method == 'Cross_layer_cross_attention':
            method_class = CrossLayerCrossAttention(seq_len, len(args.feature_layers), embed_dim=192).to(device)
        elif args.bootstrap_method == 'Last_layer':
            pass
        else:
            raise ValueError(f"Unknown bootstrap method: {args.bootstrap_method}")
    else:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    if args.use_ema:
        assert args.ema_decay < 1.0, "EMA decay should be less than 1.0"
        ema_model = EMA(copy.deepcopy(model), args.ema_decay)
        ema_model.register()

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # # following timm: set wd as 0 for bias and norm layers
    # if method_class is not None:   # 如果 method_class 存在，将其参数加入优化器
    #     param_groups_model = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    #     param_groups_method_class = optim_factory.add_weight_decay(method_class, args.weight_decay)
    #     param_groups = param_groups_model + param_groups_method_class
    #     # param_groups = param_groups_method_class
    #     # print(param_groups_method_class)
    #     # print(param_groups)
    #     # exit(0)
    # else:
    #     param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # loss_scaler = NativeScaler()

    optimizer_method_class = None
    if method_class is not None and not isinstance(method_class, FixedLayerFusion):
        param_groups_model = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        param_groups_method_class = optim_factory.add_weight_decay(method_class, args.weight_decay)

        optimizer_method_class = torch.optim.AdamW(param_groups_method_class, lr=args.lr, betas=(0.9, 0.95))
        if args.optim == "AdamW":
            # 使用 AdamW 优化器
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        elif args.optim == "SGD":
            # 使用 SGD 优化器
            optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optim == "RMSprop":
            # 使用 RMSprop 优化器
            optimizer = torch.optim.RMSprop(param_groups, lr=args.lr, alpha=0.99, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(f"Unknown optimizer: {args.optim}")
    else:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        if args.optim == "AdamW":
            # 使用 AdamW 优化器
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        elif args.optim == "SGD":
            # 使用 SGD 优化器
            optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optim == "RMSprop":
            # 使用 RMSprop 优化器
            optimizer = torch.optim.RMSprop(param_groups, lr=args.lr, alpha=0.99, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(f"Unknown optimizer: {args.optim}")

    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    if args.is_bootstrapping:
        epochs_per_bootstrap = args.epochs // args.bootstrap_steps
        remaining_epochs = args.epochs % args.bootstrap_steps
        print(f"Start training Bootstrapped MAE for {args.epochs} epochs in total, {epochs_per_bootstrap} epochs per bootstrap step")
        start_time = time.time()
        last_model = None

        for bootstrap_iter in range(args.bootstrap_steps):
            print(f"Starting bootstrap iteration {bootstrap_iter + 1}/{args.bootstrap_steps}")

            if bootstrap_iter == args.bootstrap_steps - 1:
                current_bootstrap_step_epochs = epochs_per_bootstrap + remaining_epochs
            else:
                current_bootstrap_step_epochs = epochs_per_bootstrap
            print(f"Training for {current_bootstrap_step_epochs} epochs in this bootstrap step")

            # Train for epochs_per_bootstrap epochs
            for epoch in range(current_bootstrap_step_epochs):
                if args.distributed:
                    data_loader_train.sampler.set_epoch(epoch)
                train_stats = train_one_epoch(
                    model, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    log_writer=log_writer,
                    args=args,
                    last_model=last_model,
                    method_class=method_class,
                    optimizer_method_class=optimizer_method_class,
                )
                if args.use_ema:
                    ema_model.update()
                    print("EMA model update")

                if args.bootstrap_steps > 50:   # 对于bootstrap_steps很大的情况
                    if args.output_dir and bootstrap_iter >= args.bootstrap_steps-6 and epoch + 1 == current_bootstrap_step_epochs:
                        if args.use_ema:
                            # Bmae with EMA
                            ema_model.apply_shadow()
                            misc.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, checkpoint_name=f"Bmae-ema-{bootstrap_iter + 1}")
                            ema_model.restore()
                        else:
                            # original Bmae
                            misc.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, checkpoint_name=f"Bmae-{bootstrap_iter + 1}")
                else:
                    if args.output_dir and epoch != 0 and (epoch % args.save_frequency == 0 or epoch + 1 == current_bootstrap_step_epochs):
                        if args.use_ema:
                            # Bmae with EMA
                            ema_model.apply_shadow()
                            misc.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, checkpoint_name=f"Bmae-ema-{bootstrap_iter + 1}")
                            ema_model.restore()
                        else:
                            # original Bmae
                            misc.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, checkpoint_name=f"Bmae-{bootstrap_iter + 1}")

                if args.use_ema:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                'epoch': epoch,
                                'bootstrap_iter': bootstrap_iter + 1,}
                else:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                'epoch': epoch,
                                'bootstrap_iter': bootstrap_iter + 1,}

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

            # Update target model for bootstrapping
            if args.use_ema:
                ema_model.apply_shadow()
                last_model = copy.deepcopy(ema_model.model)
                ema_model.restore()
            else:
                last_model = copy.deepcopy(model)

            # freeze last_model
            for param in last_model.parameters():
                param.requires_grad = False

            # # last_model = None
            # if args.bootstrap_method == 'Cross_layer_fusion':
            #     print(method_class.fc.weight[0][0].item())
            #     print(method_class.fc.bias[0].item())
            # elif args.bootstrap_method == 'Adaptive_layer_fusion':
            #     print(method_class.weights)   # 查看参数值是否会变化

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(f'Finish Training MAE-{bootstrap_iter + 1}. Training time {total_time_str}')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Bootstrapped MAE training time {}'.format(total_time_str))
    else:
        print(f"Start training original MAE for {args.epochs} epochs")
        start_time = time.time()

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
            if args.output_dir and epoch != 0 and (epoch % args.save_frequency == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, checkpoint_name=f"{args.name}")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Original MAE training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
