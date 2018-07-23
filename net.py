import chainer
import chainer.links as L
from chainer.functions.activation.relu import relu
import chainer.links.model.vision.resnet as R
import chainer.functions as F
from chainer.backends import cuda
from chainer.initializers import normal


def add_noise(h, sigma=1.0):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape).astype(xp.float32)
    else:
        return h


class Encoder(chainer.Chain):
    def __init__(self):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 64, **kwargs, ksize=5)
            self.bn1 = L.BatchNormalization(64, eps=1e-5)
            self.conv2 = L.Convolution2D(64, 128, **kwargs, ksize=5)
            self.bn2 = L.BatchNormalization(128, eps=1e-5)
            self.conv3 = L.Convolution2D(128, 256, **kwargs, ksize=5)
            self.bn3 = L.BatchNormalization(256, eps=1e-5)
            self.conv4 = L.Convolution2D(256, 512, **kwargs, ksize=5)
            self.bn4 = L.BatchNormalization(512, eps=1e-5)

    def __call__(self, x):
        h = relu(self.bn1(self.conv1(x)))
        h = relu(self.bn2(self.conv2(h)))
        h = relu(self.bn3(self.conv3(h)))
        h = relu(self.bn4(self.conv4(h)))

        return h


class Decoder(chainer.Chain):
    def __init__(self):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}
        with self.init_scope():
            self.deconv1 = L.Deconvolution2D(512, 256, **kwargs, ksize=5)
            self.bn1 = L.BatchNormalization(256, eps=1e-5)
            self.deconv2 = L.Deconvolution2D(256, 128, **kwargs, ksize=5)
            self.bn2 = L.BatchNormalization(128, eps=1e-5)
            self.deconv3 = L.Deconvolution2D(128, 64, **kwargs, ksize=5)
            self.bn3 = L.BatchNormalization(64, eps=1e-5)
            self.deconv4 = L.Deconvolution2D(64, 1, **kwargs, ksize=5)

    def __call__(self, x):
        h = relu(self.bn1(self.deconv1(x)))
        h = relu(self.bn2(self.deconv2(h)))
        h = relu(self.bn3(self.deconv3(h)))
        h = F.sigmoid(self.deconv4(h))

        return h


class Generator(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.enc = Encoder()
            self.dec = Decoder()

    def __call__(self, x):
        h = add_noise(x)
        h = self.enc(h)
        h = self.dec(h)

        return h


class Discriminator(chainer.Chain):
    def __init__(self):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 64, **kwargs, ksize=5)
            self.conv2 = L.Convolution2D(64, 128, **kwargs, ksize=5)
            self.bn2 = L.BatchNormalization(128, eps=1e-5)
            self.conv3 = L.Convolution2D(128, 256, **kwargs, ksize=5)
            self.bn3 = L.BatchNormalization(256, eps=1e-5)
            self.conv4 = L.Convolution2D(256, 512, **kwargs, ksize=5)
            self.bn4 = L.BatchNormalization(512, eps=1e-5)
            self.fc = L.Linear(512, 1)

    def __call__(self, x):
        h = relu(self.conv1(x))
        h = relu(self.bn2(self.conv2(h)))
        h = relu(self.bn3(self.conv3(h)))
        h = relu(self.bn4(self.conv4(h)))
        h = R._global_average_pooling_2d(h)
        h = self.fc(h)

        return h
