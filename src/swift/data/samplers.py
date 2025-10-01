import numpy as np
from torch.utils.data import BatchSampler, Sampler, Subset

# ----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.


class InfiniteSampler(Sampler):
    def __init__(
        self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5
    ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.offset = 1

    def set_offset(self, offset: int):
        """how far ahead to offset the dataset (in steps)"""
        assert isinstance(offset, int) and offset > 0, "offset must be positive"
        self.offset = offset

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.default_rng(self.seed + self.offset - 1)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                if order[i] + self.offset - 1 < order.size:
                    if self.offset > 1:
                        yield (order[i], self.offset)
                    else:
                        yield order[i]
            if window >= 2:
                j = (i - rnd.integers(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


# ----------------------------------------------------------------------------
# Wrap the sampler to give each sample in a batch the same delta


class DeltaBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: InfiniteSampler,
        batch_size: int,
        intervals: list[int],
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.intervals = intervals
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for batch in super().__iter__():
            delta = int(self.rng.choice(self.intervals))
            out = []
            for elem in batch:
                if isinstance(elem, tuple):
                    idx, offset = elem
                else:
                    idx, offset = elem, self.sampler.offset
                out.append((idx, offset, delta))
            yield out


# ----------------------------------------------------------------------------
# Subset for torch.utils.data.Dataset that allows for a subset of the dataset
# with support for attribute delegation to the original dataset, e.g., len().


class AttributeSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset

    def __getattr__(self, attr):
        """Delegate attribute access to the original dataset"""
        return getattr(self.dataset, attr)
