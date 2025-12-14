import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset, build_mySampler
from engine import evaluate, train_one_epoch, global_value
from models.detr import build as build_model


# -----------------------------
# Argument parser
# Defines all training options
# -----------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # -------------------------
    # Optimization hyperparams
    # -------------------------
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=108, type=int)
    parser.add_argument('--lr_drop', default=[72, 96], type=list)
    parser.add_argument(
        '--clip_max_norm',
        default=0.1,
        type=float,
        help='gradient clipping max norm'
    )

    # -------------------------
    # Model parameters
    # -------------------------
    parser.add_argument(
        '--frozen_weights',
        type=str,
        default=None,
        help="Path to pretrained model, only mask head will be trained if set"
    )

    # Backbone choice
    parser.add_argument(
        '--backbone',
        default='resnet50',
        type=str,
        help="Backbone network (e.g., resnet50)"
    )
    parser.add_argument(
        '--dilation',
        action='store_true',
        help="Use dilation in last conv block (DC5) instead of stride"
    )
    parser.add_argument(
        '--position_embedding',
        default='sine',
        type=str,
        help="Positional embedding type"
    )

    # -------------------------
    # Transformer settings
    # -------------------------
    parser.add_argument(
        '--enc_layers',
        default=6,
        type=int,
        help="Number of encoder layers"
    )
    parser.add_argument(
        '--dec_layers',
        default=6,
        type=int,
        help="Number of decoder layers"
    )
    parser.add_argument(
        '--dim_feedforward',
        default=2048,
        type=int,
        help="Size of feedforward layers in the transformer blocks"
    )
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int,
        help="Transformer embedding dimension"
    )
    parser.add_argument(
        '--dropout',
        default=0.1,
        type=float,
        help="Dropout rate"
    )
    parser.add_argument(
        '--nheads',
        default=8,
        type=int,
        help="Number of attention heads"
    )
    parser.add_argument(
        '--num_queries',
        default=100,
        type=int,
        help="Number of object queries"
    )
    parser.add_argument('--pre_norm', action='store_true')

    # -------------------------
    # Segmentation option
    # -------------------------
    parser.add_argument(
        '--masks',
        action='store_true',
        help="Train segmentation head when set"
    )

    # -------------------------
    # Loss and matching
    # -------------------------
    parser.add_argument(
        '--no_aux_loss',
        dest='aux_loss',
        action='store_false',
        help="Disable auxiliary losses at decoder layers"
    )

    # Matching costs for Hungarian matcher
    parser.add_argument(
        '--set_cost_class',
        default=1,
        type=float,
        help="Class cost in matching"
    )
    parser.add_argument(
        '--set_cost_bbox',
        default=5,
        type=float,
        help="L1 box cost in matching"
    )
    parser.add_argument(
        '--set_cost_giou',
        default=2,
        type=float,
        help="GIoU box cost in matching"
    )

    # Loss weighting
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument(
        '--eos_coef',
        default=0.1,
        type=float,
        help="Weight for the no-object class"
    )

    # -------------------------
    # Dataset parameters
    # -------------------------
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    # -------------------------
    # Training setup
    # -------------------------
    parser.add_argument(
        '--output_dir',
        default='',
        help='Output directory; empty means no saving'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device: "cuda" or "cpu"'
    )
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument(
        '--resume',
        default='',
        help='Checkpoint path to resume from'
    )
    parser.add_argument(
        '--load_from',
        default='',
        help='Load weights from pretrained model'
    )
    parser.add_argument(
        '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='Starting epoch index'
    )
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # -------------------------
    # Point DETR specific options
    # -------------------------
    parser.add_argument('--data_augment', action='store_true')
    parser.add_argument('--generate_pseudo_bbox', action='store_true')
    parser.add_argument('--sample_points_num', default=1, type=int)
    parser.add_argument('--save_evaltxt_forDraw', action='store_true')
    parser.add_argument(
        '--save_csv',
        default=None,
        type=str,
        help="Path to save CSV (e.g., val or trainPoint pseudo labels)"
    )

    # Consistency loss for semi-supervised training
    parser.add_argument('--cons_loss', action='store_true')
    parser.add_argument('--cons_loss_coef', default=10, type=float)

    parser.add_argument('--train_with_unlabel_imgs', action='store_true')
    parser.add_argument('--unlabel_cons_loss_coef', default=1, type=float)
    parser.add_argument('--partial', default=0, type=int)

    parser.add_argument('--start_sample_UnSupImg', default=80, type=int)
    parser.add_argument(
        '--val_data',
        default='val',
        type=str,
        help="Validation split name"
    )

    # -------------------------
    # Distributed training
    # -------------------------
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='Number of distributed processes'
    )
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='URL used to set up distributed training'
    )
    return parser


# -----------------------------
# Main training / eval function
# -----------------------------
def main(args):
    # Initialize distributed environment (rank, world_size, etc.)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    # If freezing weights, must be in segmentation mode
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    print(args)

    # Select device
    device = torch.device(args.device)

    # -------------------------
    # Fix random seeds
    # -------------------------
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Deterministic behavior for CuDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # -------------------------
    # Build model and criterion
    # -------------------------
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # Wrap model for distributed training if needed
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module

    # Count trainable parameters
    n_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print('number of params:', n_parameters)

    # -------------------------
    # Optimizer setup
    # Backbone + non-backbone groups
    # -------------------------
    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.lr_drop
    )

    # -------------------------
    # Build train and val datasets
    # -------------------------
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    print('training data:')
    print('num: ', len(dataset_train))
    print('sample: ', len(dataset_train[0]), dataset_train[0])

    # -------------------------
    # Samplers
    # -------------------------
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        # Random sampler for fully-labeled case
        if not args.train_with_unlabel_imgs:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # Custom sampler when using partial labeled data
        else:
            print('=' * 100)
            print('using MyBalancedSampler and ', str(args.partial) + '\% data.')
            print('=' * 100)
            sampler_train = build_mySampler(
                dataset_train,
                args.batch_size,
                args.partial,
                args
            )
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # -------------------------
    # Data loaders
    # -------------------------
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train,
        args.batch_size,
        drop_last=True
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers
    )

    # -------------------------
    # COCO API for evaluation
    # -------------------------
    if args.dataset_file == "coco_panoptic":
        # Also evaluate AP on original COCO during panoptic training
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    # -------------------------
    # Load frozen weights if specified
    # -------------------------
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # -------------------------
    # Resume / load checkpoints
    # -------------------------
    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume,
                map_location='cpu',
                check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])

        # Restore optimizer and scheduler when continuing training
        if (not args.eval and
            'optimizer' in checkpoint and
            'lr_scheduler' in checkpoint and
            'epoch' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.load_from:
        print('===> load from pretrained model: ', args.load_from)
        checkpoint = torch.load(args.load_from, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    # -------------------------
    # Evaluation-only mode
    # -------------------------
    if args.eval:
        test_stats, coco_evaluator = evaluate(
            args,
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir
        )
        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval,
                output_dir / "eval.pth"
            )
        return

    # -------------------------
    # Training loop
    # -------------------------
    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        # Store current epoch in global tracker
        global_value.set_value('epoch', epoch)

        # Shuffle sampler per epoch in distributed mode
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # One full training epoch
        train_stats = train_one_epoch(
            args,
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm
        )

        # Step learning rate scheduler
        lr_scheduler.step()

        # Save checkpoints
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

            # Extra checkpoints at specific epochs
            if epoch in [50, 110, 150]:
                checkpoint_paths.append(
                    output_dir / f'checkpoint{epoch:04}.pth'
                )

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    },
                    checkpoint_path
                )

        # Run evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_stats, coco_evaluator = evaluate(
                args,
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir
            )

            # Combine train and test statistics
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            # Write logs to file from main process
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # Save evaluation metrics
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                output_dir / "eval" / name
                            )

    # -------------------------
    # Print total training time
    # -------------------------
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# -----------------------------
# Script entry point
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()

    # Create output directory if needed
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
