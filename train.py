import datetime
import os
import time
import warnings

import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from torch import dropout, nn

import wandb

import argparse

import data

from sparse_token_vit import sparse_token_vit_b_16

parser = argparse.ArgumentParser(
    description="ViT training with PyTorch",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", default="resnet18", type=str, help="model name")
parser.add_argument("--token-mask", default='', type=str, help="Name of the saved token mask to train with.")
parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
parser.add_argument(
    "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
)
parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument(
    "-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers (default: 1)"
)
parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout value for fully connected layers")
parser.add_argument("--attention-dropout", default=0.0, type=float, help="Dropout value for attention")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--norm-weight-decay",
    default=None,
    type=float,
    help="weight decay for Normalization layers (default: None, same value as --wd)",
)
parser.add_argument(
    "--bias-weight-decay",
    default=None,
    type=float,
    help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
)
parser.add_argument(
    "--transformer-embedding-decay",
    default=None,
    type=float,
    help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
)
parser.add_argument(
    "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
)
parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
parser.add_argument(
    "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
)
parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
parser.add_argument("--log-freq", default=0, type=int, help="log frequency")
parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
parser.add_argument(
    "--sync-bn",
    dest="sync_bn",
    help="Use sync batch norm",
    action="store_true",
)
parser.add_argument(
    "--test-only",
    dest="test_only",
    help="Only test the model",
    action="store_true",
)
# Mixed precision training parameters
parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
# distributed training parameters
parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
parser.add_argument(
    "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
)
parser.add_argument(
    "--model-ema-steps",
    type=int,
    default=32,
    help="the number of iterations that controls how often to update the EMA model (default: 32)",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.99998,
    help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
)
parser.add_argument(
    "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
)
parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, logging=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        if logging and i % args.log_freq:
            wandb.log({'train/accuracy': acc1.item(),
                       'train/loss': loss.item()})

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", logging=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    targets = []
    outputs = []
    total_loss = 0
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            total_loss += loss.item()
            targets.append(target)
            outputs.append(output)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    
    targets = torch.cat(targets)
    outputs = torch.cat(outputs)

    total_acc1, total_acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
    if logging and log_suffix == '':
            wandb.log({'test/accuracy': total_acc1.item(),
                       'test/avg_loss': total_loss / len(data_loader)})

    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    logging = False
    if args.distributed:
        logging = torch.distributed.get_rank() == 0 and args.log_freq != 0
    else: logging = args.log_freq != 0
    if logging:
        wandb.init(project='sparse-tokens', config=args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dataloader = data.get_dataloader(torchvision.datasets.CIFAR10, batch_size=args.batch_size, num_workers=args.workers, distributed=args.distributed)
    test_dataloader = data.get_dataloader(torchvision.datasets.CIFAR10, batch_size=args.batch_size, train=False, num_workers=args.workers, distributed=args.distributed)

    num_classes = len(train_dataloader.dataset.classes)

    print("Creating model")
    if args.model != 'sparse_token_vit_b_16':
        model = torchvision.models.get_model(args.model, image_size=128, weights=args.weights, num_classes=num_classes, dropout=args.dropout, attention_dropout=args.attention_dropout)
    else:
        token_mask = torch.load(f'token_masks/{args.token_mask}/token_mask_{args.token_mask}.pt')
        model = sparse_token_vit_b_16(image_size=128, token_mask=token_mask, weights=args.weights, num_classes=num_classes, dropout=args.dropout, attention_dropout=args.attention_dropout)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, test_dataloader, device=device, log_freq=args.log_freq, log_suffix="EMA")
        else:
            evaluate(model, criterion, test_dataloader, device=device, log_freq=args.log_freq, logging=logging)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, train_dataloader, device, epoch, args, model_ema, scaler, logging)
        lr_scheduler.step()
        evaluate(model, criterion, test_dataloader, device=device, logging=logging)
        if model_ema:
            evaluate(model_ema, criterion, test_dataloader, device=device, log_freq=args.log_freq, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    if logging: wandb.finish()

if __name__ == "__main__":
    main(parser.parse_args())