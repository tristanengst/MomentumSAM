import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

from Optimizers import SAM, MSAM, NNSAM

import wandb

def get_name(args):
    suffix = f"-{args.suffix}" if not args.suffix is None else ""
    return f"{args.opt}-adapt{int(args.adaptive)}-gamma{args.gamma}-lr{args.lr}-rho{args.rho}-{args.uid}{suffix}"

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    # Utility arguments
    P.add_argument("--wandb", default="disabled", choices=["disabled", "online"],
        help="WandB usage")
    P.add_argument("--suffix", default=None, type=str,
        help="Optional suffix")
    P.add_argument("--seed", default=0, type=int,
        help="Seed")
    P.add_argument("--num_folds", default=5, type=float,
        help="Number of folds to use for cross validation. We guaruntee that for consecutive seeds that modulo NUM_FOLDS are 0 ... NUM_FOLDS - 1, the validation split is distinct. This is *not* true otherwise.")
    P.add_argument("--root", default="./data",
        help="Number of CPU threads for dataloaders.")

    # Hardware arguments
    P.add_argument("--device_id", type=int, default=0,
        help="Index of GPU to run on")
    P.add_argument("--threads", default=12, type=int,
        help="Number of CPU threads for dataloaders.")

    # Optimizer-specific arguments
    P.add_argument("--opt", choices=["sgd", "asam", "sam", "msam", "nnsam", "annsam"], required=True,
        help="optimizer")
    P.add_argument("--rho", default=0, type=float,
        help="Rho parameter for SAM.")
    P.add_argument("--gamma", type=float, default=0,
        help="gamma")
    P.add_argument("--weight_decay", default=0.0005, type=float,
        help="L2 weight decay.") 
    P.add_argument("--momentum", default=0.9, type=float,
        help="SGD Momentum.")

    # Training arguments
    P.add_argument("--batch_size", default=128, type=int,
        help="Batch size used in the training and validation loop.")
    P.add_argument("--epochs", default=200, type=int,
        help="Total number of epochs.")
    P.add_argument("--label_smoothing", default=0.1, type=float,
        help="Use 0.0 for no label smoothing.")
    P.add_argument("--lr", default=0.1, type=float,
        help="Base learning rate at the start of the training.")
    
    # Architecture arguments
    P.add_argument("--depth", default=16, type=int,
        help="Number of layers.")
    P.add_argument("--width_factor", default=8, type=int,
        help="How many times wider compared to normal ResNet.")
    P.add_argument("--dropout", default=0.0, type=float,
        help="Dropout rate.")
    args = P.parse_args()

    # Finish setting up [args]
    args.uid = wandb.util.generate_id()
    args.threads = min(args.threads, max(1, os.cpu_count() - 4))
    args.adaptive = int(args.opt.startswith("a"))
    tqdm.write(f"---------\n{args}\n---------")

    run = wandb.init(project="NNSAM",
        anonymous="allow",
        config=args,
        id=args.uid,
        name=get_name(args),
        mode=args.wandb)

    initialize(args, seed=args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(batch_size=args.batch_size,
        seed=args.seed,
        num_folds=args.num_folds,
        threads=args.threads,
        root=root)

    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout,
        in_channels=3,
        labels=10).to(device)

    base_optimizer = torch.optim.SGD

    if args.opt == "sgd":
        optimizer = SAM(model.parameters(), base_optimizer, rho=0, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "sam":
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=0, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "asam":
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=1, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "msam":
        optimizer = MSAM(model.parameters(), base_optimizer, gamma=args.gamma, rho=args.rho, adaptive=0, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "amsam":
        optimizer = MSAM(model.parameters(), base_optimizer, gamma=args.gamma, rho=args.rho, adaptive=1, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "nnsam":
        optimizer = NNSAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=0, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "annsam":
        optimizer = NNSAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=1, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = StepLR(optimizer, args.lr, args.epochs)

    for epoch in range(args.epochs):

        # Things we log for hyperparameter tuning
        losses_tr = []
        losses_te = []
        accs_te = []
        accs_val = []
        losses_val = []

        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device, non_blocking=True) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            losses_tr.append(loss.detach())

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        
        # Compute validation loss. Note that this is only logged to WandB and
        # not to the screen.
        with torch.no_grad():
            for batch in dataset.val:
                inputs, targets = (b.to(device, non_blocking=True) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets

                losses_val.append(loss)
                accs_val.append(correct)
        
        # Compute training loss
        log.eval(len_dataset=len(dataset.test))
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device, non_blocking=True) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

                losses_te.append(loss)
                accs_te.append(correct)


        # Accumulate statistics for the epoch. Note that validation and test
        # losses for a batch are returned with a reduction of 'sum', so
        # concatenating them and summing will underestimate the true value by a
        # negligible amount if the last batch of data has fewer examples than
        # the rest. However, the effect will be marginal and there will be no
        # bias in respect to a specific ordering or choice of data, and we
        # really care about accuracy anyways.
        acc_val = torch.cat(accs_val).float().mean()
        acc_te = torch.cat(accs_te).float().mean()
        loss_te = torch.cat(losses_te).cpu().mean()
        loss_val = torch.cat(losses_val).cpu().mean()
        loss_tr = torch.cat(losses_tr).mean()

        # SGD uses half the gradient evaluations of SAM-based methods.
        # Therefore, it should be run for twice as long. This will give a slight
        # advantage in that not only does it get twice as many queries to the
        # first order oracle, but it can see stochastic data twice as much.
        grad_evals = epoch * len(dataset.train) * (1 if args.opt == "sgd" else 2)
        wandb.log({"gradient_evals": grad_evals,
            "loss/te": loss_te,
            "acc/te": acc_te,
            "acc/val": acc_val,
            "loss/tr": loss_tr,
            "loss_val": loss_val,
            "epoch": epoch})

    log.flush()
