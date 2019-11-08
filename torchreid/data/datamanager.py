import torch

from .datasets import Market1501
from .datasetloader import ImageDataset
from .sampler import build_train_sampler
from .transforms import build_transforms


class ImageDataManager(object):

    def __init__(self, height=256, width=128, transforms='random_flip', norm_mean=None, norm_std=None,
                 batch_size_train=32, batch_size_test=32, num_instances=4, train_sampler=''):

        transform_tr, transform_te = build_transforms(height, width, transforms=transforms,
                                                      norm_mean=norm_mean, norm_std=norm_std)

        print('=> Loading dataset')
        dataset = Market1501()
        self.num_train_pids = dataset.num_train_pids

        train_sampler = build_train_sampler(dataset.train, train_sampler, batch_size=batch_size_train,
                                            num_instances=num_instances)

        self.trainloader = torch.utils.data.DataLoader(
            ImageDataset(dataset.train, transform=transform_tr),
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        self.queryloader = torch.utils.data.DataLoader(
            ImageDataset(dataset.query, transform=transform_te),
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        self.galleryloader = torch.utils.data.DataLoader(
            ImageDataset(dataset.gallery, transform=transform_te),
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        self.querydataset = dataset.query
        self.gallerydataset = dataset.gallery

    def return_dataloaders(self):
        return self.trainloader, self.queryloader, self.galleryloader

    def return_testdataset(self):
        return self.querydataset, self.gallerydataset
