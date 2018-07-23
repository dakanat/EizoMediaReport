import argparse
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
import chainer.functions as F

from net import Generator, Discriminator
from updater import AdversarialUpdater
from visualize import out_generated_image, in_generated_image
from dataset import load_mnist


def main():
    parser = argparse.ArgumentParser(description='Chainer Novelty Detection')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--digit', '-d', type=int, default=0,
                        help='digit number as one class')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--snapshot_interval', type=int, default=10,
                        help='Interval of snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    gen = Generator()
    dis = Discriminator()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    def make_optimizer(model, alpha=1e-5, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(
            chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    train, test = load_mnist(args.digit)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = AdversarialUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'epoch')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport([
        'epoch', 'gen/loss', 'gen/l1', 'gen/l2', 'dis/loss', 'dis/l1', 'dis/l2',
    ]))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            test, gen,
            10, 10, args.gpu, args.out),
        trigger=(10, 'epoch'))
    trainer.extend(
        in_generated_image(
            test, gen,
            10, 10, args.gpu, args.out),
        trigger=(10, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
