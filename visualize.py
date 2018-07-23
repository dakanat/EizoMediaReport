import os

import numpy as np
from PIL import Image

import chainer
import chainer.backends.cuda
from chainer import Variable


def out_generated_image(data, gen, rows, cols, gpu, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_images = rows * cols
        xp = gen.xp
        x, t = chainer.dataset.concat_examples(data[len(data)-n_images:], device=gpu)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(x)
        x = chainer.backends.cuda.to_cpu(x.data)

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 1, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 1))
        x = x.transpose(2, 0, 1)[0]

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/out{}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image

def in_generated_image(data, gen, rows, cols, gpu, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_images = rows * cols
        xp = gen.xp
        x, t = chainer.dataset.concat_examples(data[:n_images], device=gpu)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(x)
        x = chainer.backends.cuda.to_cpu(x.data)

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 1, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 1))
        x = x.transpose(2, 0, 1)[0]

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/in{}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image
