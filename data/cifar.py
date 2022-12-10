from tqdm import tqdm
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from utility.cutout import Cutout


class Cifar:
    def __init__(self, batch_size=128, seed=0, num_folds=0, threads=12, root="./data"):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.train_set = torchvision.datasets.CIFAR10(root=root,
            train=True,
            download=True,
            transform=train_transform)
        self.val_set = torchvision.datasets.CIFAR10(root=root,
            train=True,
            download=True,
            transform=test_transform)
        self.test_set = torchvision.datasets.CIFAR10(root=root,
            train=False,
            download=True,
            transform=test_transform)

        self.seed = seed
        if num_folds == 0:
            # In this case, we train on all the training data and validate on
            # all the training data. Note that the augmentations are much
            # stronger when training than augmentation even though the
            # underlying data is the same.
            self.fold_idx = -1
            self.idxs_tr = range(len(self.train_set))
            self.idxs_val = range(len(self.val_set))
        else:
            # In this case, we will contribute to K-fold cross validation. We
            # guaruntee that for sequential seeds that modulo NUM_FOLDS are
            # 0 ... NUM_FOLDS - 1, the validation splits will be distinct, and
            # that with high probability this is not true otherwise.
            if not (len(self.train_set) / num_folds).is_integer():
                raise ValueError(f"The number of folds should evenly divide the length of the dataset. Got {num_folds} folds and {len(self.train_set)} training examples")

            # Add one to deal with the case in which the dataset isn't shuffled.
            # We need to set a different seed here so that two seeds that give
            # the same number of reshufflings have the same seed for this code.
            num_reshufflings = 1 + abs(seed // num_folds)
            all_idxs = range(len(self.train_set))
            
            random.seed(seed // num_folds)
            for _ in tqdm(range(num_reshufflings),
                desc="Reshuffling data",
                leave=False,
                dynamic_ncols=True):

                all_idxs = random.sample(all_idxs, k=len(all_idxs))
            random.seed(seed)

            # Splice the datasets
            self.fold_idx = seed % num_folds
            num_val_idxs = len(self.train_set) // num_folds
            self.idxs_val = all_idxs[num_val_idxs * self.fold_idx:num_val_idxs * (self.fold_idx+1)]
            idxs_val_set = set(self.idxs_val)
            self.idxs_tr = [idx for idx in all_idxs if not idx in idxs_val_set]

            self.train_set = Subset(self.train_set, indices=self.idxs_tr)
            self.val_set = Subset(self.val_set, indices=self.idxs_val)

        self.val = torch.utils.data.DataLoader(self.val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=threads,
            pin_memory=True)

        self.train = torch.utils.data.DataLoader(self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=threads,
            pin_memory=True,
            persistent_workers=True)

        self.test = torch.utils.data.DataLoader(self.test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=threads,
            pin_memory=True)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        tqdm.write(str(self))

    def __str__(self): return f"{self.__class__.__name__} [fold_index={self.fold_idx} num_train_ex={len(self.train_set)} num_val_ex={len(self.val_set)} seed={self.seed}]"

    def _get_statistics(self, root="./data"):
        train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
