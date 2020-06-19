import gunpowder as gp
import gunpowder.torch as gp_torch
import math
import numpy as np
import torch
import logging
from model import SpineUNet

logging.basicConfig(level=logging.INFO)


class AddChannelDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        if self.array not in batch:
            return

        batch[self.array].data = batch[self.array].data[np.newaxis]


class RemoveChannelDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        if self.array not in batch:
            return

        batch[self.array].data = batch[self.array].data[0]


def train(until):

    model = SpineUNet()
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_size = (8, 96, 96)

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    affs = gp.ArrayKey('AFFS')
    affs_predicted = gp.ArrayKey('AFFS_PREDICTED')

    pipeline = (
        (
            gp.ZarrSource(
                'data/20200201.zarr',
                {
                    raw: 'train/sample1/raw',
                    labels: 'train/sample1/labels'
                }),
            gp.ZarrSource(
                'data/20200201.zarr',
                {
                    raw: 'train/sample2/raw',
                    labels: 'train/sample2/labels'
                }),
            gp.ZarrSource(
                'data/20200201.zarr',
                {
                    raw: 'train/sample3/raw',
                    labels: 'train/sample3/labels'
                })
        ) +
        gp.RandomProvider() +
        gp.Normalize(raw) +
        gp.RandomLocation() +
        gp.SimpleAugment(transpose_only=(1, 2)) +
        gp.ElasticAugment((2, 10, 10), (0.0, 0.5, 0.5), [0, math.pi]) +
        gp.AddAffinities(
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            labels,
            affs) +
        gp.Normalize(affs, factor=1.0) +
        #gp.PreCache(num_workers=1) +
        # raw: (d, h, w)
        # affs: (3, d, h, w)
        gp.Stack(1) +
        # raw: (1, d, h, w)
        # affs: (1, 3, d, h, w)
        AddChannelDim(raw) +
        # raw: (1, 1, d, h, w)
        # affs: (1, 3, d, h, w)
        gp_torch.Train(
            model,
            loss,
            optimizer,
            inputs={'x': raw},
            outputs={0: affs_predicted},
            loss_inputs={0: affs_predicted, 1: affs},
            save_every=10000) +
        RemoveChannelDim(raw) +
        RemoveChannelDim(raw) +
        RemoveChannelDim(affs) +
        RemoveChannelDim(affs_predicted) +
        # raw: (d, h, w)
        # affs: (3, d, h, w)
        # affs_predicted: (3, d, h, w)
        gp.Snapshot(
            {
                raw: 'raw',
                labels: 'labels',
                affs: 'affs',
                affs_predicted: 'affs_predicted'
            },
            every=500,
            output_filename='iteration_{iteration}.hdf')
    )

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, input_size)
    request.add(affs, input_size)
    request.add(affs_predicted, input_size)

    with gp.build(pipeline):
        for i in range(until):
            pipeline.request_batch(request)
#%%
if __name__ == "__main__":
    train(100000)
