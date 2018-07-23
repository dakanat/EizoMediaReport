import numpy as np
import chainer


class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, digit):
        super().__init__()
        self.dataset = dataset
        self.digit = digit

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        image, label = self.dataset[i]
        if label == self.digit:
            label = np.ones(1, dtype=np.int32)[0]
        else:
            label = np.zeros(1, dtype=np.int32)[0]

        return image, label


def load_mnist(digit, percent=0.5):
    train_data, test_data = chainer.datasets.get_mnist(ndim=3)
    train_key = np.where(np.array(train_data)[:, 1] == digit)[0]
    test_key1 = np.where(np.array(test_data)[:, 1] == digit)[0]
    test_key2 = np.where(np.array(test_data)[:, 1] != digit)[0][:int(len(test_key1)/(1-percent)*percent)]

    train = Dataset(np.array(train_data)[train_key], digit)
    test = Dataset(chainer.datasets.ConcatenatedDataset(np.array(test_data)[test_key1], np.array(test_data)[test_key2]), digit)

    return train, test
