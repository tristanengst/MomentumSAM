import argparse
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

import sys; sys.path.append("..")
from sam import SAM

import wandb

def get_name(args):
    suffix = f"-{arg.suffix}" if not args.suffix is None else ""
    return f"{args.opt}-adapt{int(args.adaptive)}-gamma{args.gamma}-lr{args.lr}-rho{args.rho}-{args.uid}{suffix}"

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--wandb", default="disabled", choices=["disabled", "online"],
        help="WandB usage")
    P.add_argument("--opt", choices=["sgd", "sam", "msam"], required=True,
        help="optimizer")
    P.add_argument("--adaptive", default=0, type=int, choices=[0, 1],
        help="True if you want to use the Adaptive SAM.")
    P.add_argument("--gamma", type=float, default=.1,
        help="gamma")
    P.add_argument("--batch_size", default=128, type=int,
        help="Batch size used in the training and validation loop.")
    P.add_argument("--depth", default=16, type=int,
        help="Number of layers.")
    P.add_argument("--dropout", default=0.0, type=float,
        help="Dropout rate.")
    P.add_argument("--epochs", default=200, type=int,
        help="Total number of epochs.")
    P.add_argument("--label_smoothing", default=0.1, type=float,
        help="Use 0.0 for no label smoothing.")
    P.add_argument("--lr", default=0.1, type=float,
        help="Base learning rate at the start of the training.")
    P.add_argument("--momentum", default=0.9, type=float,
        help="SGD Momentum.")
    P.add_argument("--threads", default=20, type=int,
        help="Number of CPU threads for dataloaders.")
    P.add_argument("--rho", default=.05, type=float,
        help="Rho parameter for SAM.")
    P.add_argument("--weight_decay", default=0.0005, type=float,
        help="L2 weight decay.")
    P.add_argument("--width_factor", default=8, type=int,
        help="How many times wider compared to normal ResNet.")
    P.add_argument("--suffix", default=None, type=str,
        help="Optional suffix")
    args = P.parse_args()
    args.uid = wandb.util.generate_id()
    tqdm.write(str(args))

    run = wandb.init(project="MomentumSAM",
        anonymous="allow",
        config=args,
        id=args.uid,
        name=get_name(args),
        mode=args.wandb)

    initialize(args, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    base_optimizer = torch.optim.SGD

    if args.opt == "sgd":
        optimizer = SAM(model.parameters(), base_optimizer, rho=0, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "sam":
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "msam":
        optimizer = MSAM(model.parameters(), base_optimizer, gamma=args.gamma, rho=args.rho, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.lr, args.epochs)

    for epoch in range(args.epochs):

        # Things we log for hyperparameter tuning
        losses_tr = []
        losses_te = []
        accs_te = []

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


        acc_te = torch.cat(accs_te).float().mean()
        loss_te = torch.cat(losses_te).cpu().mean()
        loss_tr = torch.cat(losses_tr).mean()

        wandb.log({"epoch": epoch,
            "loss/te": loss_te,
            "acc/te": acc_te,
            "loss_tr": loss_tr})


    log.flush()
