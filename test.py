import argparse
import numpy as np
import os
import sys
import chainer
import chainer.functions as F

from net import Generator, Discriminator
from dataset import load_mnist


def main():
    parser = argparse.ArgumentParser(description='Chainer Novelty Detection')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--percent', '-p', type=float, default=0.5,
                        help='percentage of outliers')
    parser.add_argument('--threshold', '-t', type=float, default=0.4,
                        help='threshold of outliers')
    parser.add_argument('--load_epoch', type=str, default='30')
    args = parser.parse_args()

    print(args)


    gen = Generator()
    dis = Discriminator()

    D = []
    DR = []
    print('\tdigit\tpre\trec\tf1_score')

    for i in range(10):
        if os.path.exists(f'result_{i}/gen_epoch_{args.load_epoch}.npz'):
            chainer.serializers.load_npz(f'result_{i}/gen_epoch_{args.load_epoch}.npz', gen)
        else:
            sys.exit(f'result_{i}/gen_epoch_{args.load_epoch}.npz does not exist.')

        if os.path.exists(f'result_{i}/dis_epoch_{args.load_epoch}.npz'):
            chainer.serializers.load_npz(f'result_{i}/dis_epoch_{args.load_epoch}.npz', dis)
        else:
            sys.exit(f'result_{i}/dis_epoch_{args.load_epoch}.npz does not exist.')

        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            gen.to_gpu()
            dis.to_gpu()

        train, test = load_mnist(i, args.percent)
        test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

        outputs = []
        outputs_dash = []
        labels = []
        for batch in test_iter:
            x, t = chainer.dataset.concat_examples(batch, device=args.gpu)
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                y = F.sigmoid(dis(x))
                x_dash = gen(x)
                y_dash = F.sigmoid(dis(x_dash))
            outputs.extend(y.data[:, 0])
            outputs_dash.extend(y_dash.data[:, 0])
            labels.extend(t)

        outputs = np.array(outputs)
        outputs_dash = np.array(outputs_dash)
        labels = np.array(labels)

        positive = np.where(outputs > args.threshold)[0]
        tp = np.sum(labels[positive] == 1)
        fp = np.sum(labels[positive] == 0)
        negative = np.where(outputs <= args.threshold)[0]
        tn = np.sum(labels[negative] == 0)
        fn = np.sum(labels[negative] == 1)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * recall * precision / (recall + precision + 1e-8)
        print('D(X)\t{:d}\t{:4.3f}\t{:4.3f}\t{:4.3f}'.format(i, precision, recall, f1_score))
        D.append(f1_score)
        positive = np.where(outputs_dash > args.threshold)[0]
        tp = np.sum(labels[positive] == 1)
        fp = np.sum(labels[positive] == 0)
        negative = np.where(outputs_dash <= args.threshold)[0]
        tn = np.sum(labels[negative] == 0)
        fn = np.sum(labels[negative] == 1)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * recall * precision / (recall + precision + 1e-8)
        print('D(R(X))\t{:d}\t{:4.3f}\t{:4.3f}\t{:4.3f}'.format(i, precision, recall, f1_score))
        DR.append(f1_score)

    print('D(X) f1_score {:4.3f}'.format(np.mean(np.array(D))))
    print('D(R(X)) f1_score {:4.3f}'.format(np.mean(np.array(DR))))



if __name__ == '__main__':
    main()
