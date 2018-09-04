import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import theano
from theano import tensor as T
import numpy as np
import time
from scipy import misc

import neuralnet as nn

srng = theano.sandbox.rng_mrg.MRG_RandomStreams(np.random.randint(1, int(time.time())))


class LSGAN(nn.Sequential, nn.model_zoo.Net):
    def __init__(self, input_shape, output_size):
        super(LSGAN, self).__init__(input_shape=input_shape, layer_name='LSGAN')
        s2, s4, s8, s16 = output_size // 2, output_size // 4, output_size // 8, output_size // 16

        subnet = 'generator'
        self.gen = nn.Sequential(input_shape=input_shape, layer_name=subnet)
        self.gen.append(nn.FullyConnectedLayer(self.gen.output_shape, 256 * s16 * s16, activation='linear',
                                               layer_name=subnet + '/fc1', init=nn.Normal(.02)))
        self.gen.append(nn.ReshapingLayer(self.gen.output_shape, (-1, 256, s16, s16), subnet+'/reshape'))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, subnet + '/bn1', activation='relu', epsilon=1.1e-5))

        self.gen.append(nn.TransposedConvolutionalLayer(self.gen.output_shape, 256, 3, (s8, s8), activation='linear',
                                                        layer_name=subnet + '/deconv1', init=nn.Normal(.02)))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, subnet + '/bn2', activation='relu', epsilon=1.1e-5))
        self.gen.append(
            nn.TransposedConvolutionalLayer(self.gen.output_shape, 256, 3, (s8, s8), stride=(1, 1), activation='linear',
                                            layer_name=subnet + '/deconv2', init=nn.Normal(.02)))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, subnet + '/bn3', activation='relu', epsilon=1.1e-5))

        self.gen.append(nn.TransposedConvolutionalLayer(self.gen.output_shape, 256, 3, (s4, s4), activation='linear',
                                                        layer_name=subnet + '/deconv3', init=nn.Normal(.02)))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, subnet + '/bn4', activation='relu', epsilon=1.1e-5))
        self.gen.append(
            nn.TransposedConvolutionalLayer(self.gen.output_shape, 256, 3, (s4, s4), stride=(1, 1), activation='linear',
                                            layer_name=subnet + '/deconv4', init=nn.Normal(.02)))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, subnet + '/bn5', activation='relu', epsilon=1.1e-5))

        self.gen.append(nn.TransposedConvolutionalLayer(self.gen.output_shape, 128, 3, (s2, s2), activation='linear',
                                                        layer_name=subnet + '/deconv5', init=nn.Normal(.02)))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, subnet + '/bn6', activation='relu', epsilon=1.1e-5))

        self.gen.append(nn.TransposedConvolutionalLayer(self.gen.output_shape, 64, 3, (output_size, output_size),
                                                        activation='linear', layer_name=subnet + '/deconv6',
                                                        init=nn.Normal(.02)))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, subnet + '/bn7', activation='relu', epsilon=1.1e-5))

        self.gen.append(
            nn.TransposedConvolutionalLayer(self.gen.output_shape, 3, 3, (output_size, output_size), stride=(1, 1),
                                            activation='tanh', layer_name=subnet + '/output', init=nn.Normal(.02)))
        self.append(self.gen)

        subnet = 'discriminator'
        self.dis = nn.Sequential(input_shape=self.gen.output_shape, layer_name=subnet)
        self.dis.append(nn.ConvolutionalLayer(self.dis.output_shape, 64, 5, stride=2, activation='lrelu', no_bias=False,
                                              layer_name=subnet + '/first_conv', init=nn.TruncatedNormal(.02), alpha=.2))
        self.dis.append(nn.ConvNormAct(self.dis.output_shape, 64*2, 5, stride=2, activation='lrelu', no_bias=False,
                                       layer_name=subnet + '/conv1', init=nn.TruncatedNormal(.02), epsilon=1.1e-5, alpha=.2))
        self.dis.append(nn.ConvNormAct(self.dis.output_shape, 64*4, 5, stride=2, activation='lrelu', no_bias=False,
                                       layer_name=subnet + '/conv2', init=nn.TruncatedNormal(.02), epsilon=1.1e-5, alpha=.2))
        self.dis.append(nn.ConvNormAct(self.dis.output_shape, 64*8, 5, stride=2, activation='lrelu', no_bias=False,
                                       layer_name=subnet + '/conv3', init=nn.TruncatedNormal(.02), epsilon=1.1e-5, alpha=.2))
        self.dis.append(
            nn.FullyConnectedLayer(self.dis.output_shape, 1, layer_name=subnet + '/output', activation='linear'))
        self.append(self.dis)

    def get_output(self, input):
        return self.gen(input)

    @property
    def output_shape(self):
        return self.dis.output_shape

    def get_cost(self, image, noise):
        fake = self.gen(noise)
        pred_real = self.dis(image)
        pred_fake = self.dis(fake)

        gen_loss = T.sum((pred_fake - 1.) ** 2. / 2.)
        dis_loss_real = T.sum((pred_real - 1.) ** 2. / 2.)
        dis_loss_fake = T.sum(pred_fake ** 2. / 2.)
        dis_loss = dis_loss_real + dis_loss_fake
        return gen_loss, dis_loss

    def learn(self, image, noise, lr, beta1):
        gen_cost, dis_cost = self.get_cost(image, noise)
        self.opt_gen, updates_gen = nn.adam(gen_cost, self.gen.trainable, lr, beta1, return_op=True)
        self.opt_dis, updates_dis = nn.adam(dis_cost, self.dis.trainable, lr, beta1, return_op=True)
        return gen_cost, dis_cost, updates_gen, updates_dis


class DataManager(nn.DataManager):
    def __init__(self, output_size, placeholders, path, batchsize, n_epochs, shuffle):
        super(DataManager, self).__init__(None, placeholders, path=path, batch_size=batchsize, n_epochs=n_epochs,
                                          shuffle=shuffle)
        self.output_size = output_size
        self.load_data()

    def load_data(self):
        self.dataset = os.listdir(self.path)
        self.data_size = len(self.dataset)

    def generator(self):
        num_batches = self.data_size // self.batch_size
        dataset = self.dataset
        if self.shuffle:
            from random import shuffle
            shuffle(dataset)
        for i in range(num_batches):
            batch = dataset[i * self.batch_size:(i + 1) * self.batch_size]
            batch = np.array(
                [misc.imresize(misc.imread(self.path + '/' + file), (self.output_size, self.output_size)) for file in
                 batch], 'float32')
            yield batch

    @staticmethod
    def preprocess(input):
        return input.dimshuffle(0, 3, 1, 2) / 127.5 - 1.


def train(input_shape, output_size, image_path, bs=64, z_dim=1024, n_iters=int(1e6)):
    batch_input_shape = (bs,) + input_shape
    net = LSGAN((bs, z_dim), output_size)

    X__ = T.tensor4('image', 'float32')
    X = DataManager.preprocess(X__)
    z = srng.uniform((bs, z_dim), -1, 1, ndim=2, dtype='float32')
    X_ = theano.shared(np.zeros(batch_input_shape, 'float32'), 'image placeholder')

    nn.set_training_status(True)
    gen_loss, dis_loss, updates_gen, updates_dis = net.learn(X, z, 1e-3, .5)
    train_gen = nn.function([], gen_loss, updates=updates_gen, name='train generator')
    train_dis = nn.function([], dis_loss, updates=updates_dis, givens={X__: X_}, name='train discriminator')

    nn.set_training_status(False)
    fixed_noise = T.constant(np.random.uniform(-1, 1, (bs, z_dim)), 'fixed noise', 2, 'float32')
    gen_imgs = net(fixed_noise)
    generate = nn.function([], gen_imgs, name='generate images')

    n_epochs = int(bs * n_iters / len(os.listdir(image_path)))
    dm = DataManager(output_size, X_, image_path, bs, n_epochs, True)
    mon = nn.monitor.Monitor(model_name='LSGAN', use_visdom=True)
    batches = dm.get_batches()
    start = time.time()
    for it in batches:
        dis_loss_ = train_dis()
        gen_loss_ = train_gen()
        if np.isnan(gen_loss_ + dis_loss_) or np.isinf(dis_loss_ + gen_loss_):
            raise ValueError('Training failed! Stopped.')
        mon.plot('discriminator loss', dis_loss_)
        mon.plot('generator loss', gen_loss_)

        if it % 500 == 499:
            fake_imgs = generate()
            mon.imwrite('fake images', fake_imgs / 2. + .5)
            mon.plot('time elapsed', (time.time() - start)/60.)
            nn.utils.save(net, mon.current_folder + '/training.pkl')
            mon.flush()
        mon.tick()
    net.save_params(mon.current_folder + '/params.npz')
    print('Training finished')


if __name__ == '__main__':
    train((112, 112, 3), 112, 'D:/1_Share/LSUN/church_outdoor_train_lmdb/data')
