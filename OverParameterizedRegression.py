
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import wandb
import os
import itertools

from utility.step_lr import StepLR
from utility.initialize import initialize
from Optimizers import SAM, MSAM, NNSAM

def compute_eval_loss(model, eval_loader):
    loss = 0
    total = 0
    for idx,batch in tqdm(enumerate(eval_loader),
        desc="Validation",
        dynamic_ncols=True,
        leave=False):

        inputs, targets = (b.to(device, non_blocking=True) for b in batch)
        loss += model(inputs, targets) * batch[0].shape[0]
        total += batch[0].shape[0]
    return loss.item() / total

def get_name(args):
    suffix = f"-{args.suffix}" if not args.suffix is None else ""
    return f"matrix-{args.opt}-lr{args.lr}-rho{args.rho}-wd{args.weight_decay}-{args.uid}{suffix}"


class QuadraticProblemDataset(Dataset):

    def __init__(self, x_dim=100, y_dim=100, n=10000, noise_std_frac=.1):
        super(QuadraticProblemDataset, self).__init__()
        # if not x_dim > n:
        #     raise ValueError(f"Matrix dimension (x_dim) was {x_dim} but should be greater than the number of examples {n}")
        
        self.x_dim = x_dim
        self.n = n
        self.y_dim = y_dim
        
        # We want AA.T to be full rank, following the paper. This is VERY likely
        # when we generate the matrix as follows, but the while loop throws an error.
        self.A = torch.randn(x_dim, y_dim, device=device)
        # while torch.linalg.matrix_rank(self.A @ self.A.transpose(1, 0)) < y_dim:
        #     self.A = torch.randn(x_dim, y_dim, device=device)

        self.X = torch.randn(n, x_dim, device=device)
        self.Y = self.X @ self.A

        y_std = torch.diag(torch.std(self.Y, dim=0) * noise_std_frac)
        y_noise = torch.randn(*self.Y.shape, device=device) @ y_std
        self.Y = self.Y + y_noise

        self.A = self.A.cpu()
        self.X = self.X.cpu()
        self.Y = self.Y.cpu()

    def __repr__(self): return f"{self.__class__.__name__} [x_dim={self.x_dim} y_dim={self.y_dim} length={self.n}]"

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

    def compute_loss(self, A):
        with torch.no_grad():
            return torch.mean(torch.sum(torch.square(self.X @ A - self.Y), dim=1))

class QuadraticModel(nn.Module):

    def __init__(self, x_dim=100, y_dim=100):
        super(QuadraticModel, self).__init__()
        self.A = nn.Parameter(torch.randn(x_dim, y_dim))

    def forward(self, x, y):
        return torch.mean(torch.sum(torch.square(x @ self.A - y), dim=1))
    
    def __repr__(self): return f"{self.__class__.__name__} [x_dim={self.x_dim} y_dim={self.y_dim}]"

def get_args():
    P = argparse.ArgumentParser()
    P.add_argument("--wandb", default="disabled", choices=["disabled", "online"],
        help="WandB usage")
    P.add_argument("--suffix", default=None, type=str,
        help="Optional suffix")
    P.add_argument("--seed", default=0, type=int,
        help="Seed")
    P.add_argument("--eval_iter", default=100, type=int,
        help="Evaluate every EVAL_ITER steps")
    P.add_argument("--log_iter", default=100, type=int,
        help="Log every LOG_ITER steps")
    P.add_argument("--train_ex", default=1, type=int)
    P.add_argument("--val_ex", default=1, type=int)
    P.add_argument("--test_ex", default=1, type=int)
    P.add_argument("--y_dim", default=100, type=int)
    P.add_argument("--x_dim", default=100, type=int)
    P.add_argument("--lr", default=1e-3, type=float)
    P.add_argument("--momentum", default=0, type=float,
        help="SGD Momentum.")
    P.add_argument("--noise_std_frac", default=.1, type=float,
        help="SGD Momentum.")
    P.add_argument("--num_steps", default=1000, type=int,
        help="SGD Momentum.")
    P.add_argument("--weight_decay", default=0, type=float)
    P.add_argument("--batch_size", default=100, type=int)
    P.add_argument("--opt", choices=["sgd", "sam", "nnsam"], required=True)
    P.add_argument("--rho", default=1, type=float)
    P.add_argument("--threads", default=12, type=int,
        help="Number of CPU threads for dataloaders.")
    P.add_argument("--device_id", type=int, default=0,
        help="Index of GPU to run on")
    args = P.parse_args()
    args.uid = wandb.util.generate_id()
    args.threads = min(args.threads, max(1, os.cpu_count() - 4))
    args.sub_problem = "matrix"
    return args

if __name__ == "__main__":
    args = get_args()
    initialize(args, seed=args.seed)
    # device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    run = wandb.init(project="NNSAM",
        anonymous="allow",
        config=args,
        id=args.uid,
        name=get_name(args),
        mode=args.wandb)

    data = QuadraticProblemDataset(x_dim=args.x_dim,
        y_dim=args.y_dim,
        noise_std_frac=args.noise_std_frac,
        n=args.train_ex + args.val_ex + args.test_ex)
    tqdm.write(str(data))
    data_tr = Subset(data, indices=range(args.train_ex))
    data_val = Subset(data, indices=range(args.train_ex, args.val_ex + args.train_ex))
    data_te = Subset(data, indices=range(args.val_ex + args.train_ex, args.train_ex + args.val_ex + args.test_ex))

    loader_tr = DataLoader(data_tr,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=args.threads)
    loader_val = DataLoader(data_val,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=args.threads)
    loader_te = DataLoader(data_te,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=args.threads)

    model = QuadraticModel(x_dim=args.x_dim, y_dim=args.y_dim).to(device)
    base_optimizer = torch.optim.SGD

    if args.opt == "sgd":
        optimizer = SAM(model.parameters(), base_optimizer, rho=0, adaptive=0, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
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

    passes_over_dataset = args.num_steps // len(loader_tr)
    args.num_steps = passes_over_dataset * len(loader_tr)
    chain_loader = itertools.chain(*[loader_tr] * passes_over_dataset)
    scheduler = StepLR(optimizer, args.lr, passes_over_dataset)

    tqdm.write(str(args))

    for idx,batch in tqdm(enumerate(chain_loader),
        desc="Iterations",
        dynamic_ncols=True,
        total=args.num_steps):

        inputs, targets = (b.to(device, non_blocking=True) for b in batch)

        # first forward-backward step
        loss = model(inputs, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        _ = model(inputs, targets).backward()
        optimizer.second_step(zero_grad=True)

        if idx % args.eval_iter == 0 or idx == args.num_steps - 1:
            loss_val = compute_eval_loss(model, loader_val)
            loss_te = compute_eval_loss(model, loader_te)
            tqdm.write(f"Step {idx}/{args.num_steps} - loss_tr={loss.item():.5f} loss_val={loss_val:.5f} loss_te={loss_te:.5f}")
            wandb.log({"cur_step": idx, "loss/tr": loss, "loss/te": loss_te, "loss/val": loss_val})

        elif idx % args.log_iter == 0 or idx == args.num_steps - 1:
            tqdm.write(f"Step {idx}/{args.num_steps} - loss_tr={loss.item():.5f}")
            wandb.log({"cur_step": idx, "loss/tr": loss})

        scheduler(idx // passes_over_dataset)
