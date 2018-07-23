import chainer
import chainer.functions as F
from chainer import Variable


class AdversarialUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super().__init__(*args, **kwargs)

    def loss_dis(self, y, y_dash):
        loss1 = F.sum(F.softplus(-y)) / len(y)
        loss2 = F.sum(F.softplus(y_dash)) / len(y_dash)
        loss = loss1 + loss2
        chainer.report({'loss': loss, 'l1': loss1, 'l2': loss2}, self.dis)

        return loss

    def loss_gen(self, x, x_dash, y_dash):
        loss1 = F.mean_squared_error(x, x_dash)
        loss2 = F.sum(F.softplus(-y_dash)) / len(y_dash)
        loss = 0.4 * loss1 + loss2
        chainer.report({'loss': loss, 'l1': loss1, 'l2': loss2}, self.gen)

        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        in_arrays = self.converter(batch, self.device)

        x, t = in_arrays
        x = Variable(x)

        y = self.dis(x)
        x_dash = self.gen(x)
        y_dash = self.dis(x_dash)

        dis_optimizer.update(self.loss_dis, y, y_dash)
        gen_optimizer.update(self.loss_gen, x, x_dash, y_dash)
