import os
import argparse
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils.generic_utils import Progbar
import numpy as np
import utils
import models

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--result_root', default=os.path.join(curdir, 'results'))
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--snap_freq', type=int, default=5)
parser.add_argument('--g_lr', type=float, default=2e-4)
parser.add_argument('--d_lr', type=float, default=1e-5)
parser.add_argument('--g_beta1', type=float, default=0.5)
parser.add_argument('--d_beta1', type=float, default=0.1)

def main(args):

    # ===========================
    # Create result root(if necessary)
    # ===========================
    if os.path.exists(args.result_root) == False:
        os.makedirs(args.result_root)

    # ===========================
    # Prepare datasets
    # ===========================
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = utils.preprocess_input(x_train)

    # ===========================
    # Instantiate models
    # ===========================
    generater = models.create_generater()
    discriminater = models.create_discriminater()
    dcgan = Sequential([generater, discriminater])

    discriminater.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=args.d_lr, beta_1=args.d_beta1))

    discriminater.trainable = False
    dcgan.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=args.g_lr, beta_1=args.g_beta1))

    # ===========================
    # Train models
    # ===========================
    for epoch in range(args.epochs):

        # create progress bar
        progbar = Progbar(len(x_train))

        # shuffle dataset
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]

        for i in range(0, len(x_train), args.batch_size):
            z_d = np.random.uniform(-1,1,size=(args.batch_size, 100))
            z_g = np.random.uniform(-1,1,size=(args.batch_size, 100))
            real = x_train[i:i+args.batch_size]
            fake = generater.predict_on_batch(z_d)

            # train discriminater
            x = np.concatenate((real,fake), axis=0)
            y = [1.]*args.batch_size + [0.]*args.batch_size
            d_loss = discriminater.train_on_batch(x,y)

            # train generater
            x = z_g
            y = [1.]*args.batch_size
            g_loss = dcgan.train_on_batch(x,y)

            # update progress bar
            progbar.add(len(real), values=[
                ('epoch', epoch+1),
                ('d_loss', d_loss),
                ('g_loss', g_loss),
            ])

        if (epoch+1) % args.snap_freq == 0:
            savedir = os.path.join(args.result_root, 'epoch%d' % (epoch+1))
            if os.path.exists(savedir) == False:
                os.mkdir(savedir)

            # save models' weights
            generater.save_weights(os.path.join(savedir, 'generater_weights.h5'))
            discriminater.save_weights(os.path.join(savedir, 'discriminater_weights.h5'))

            # save generated image
            img = utils.create_generated_img(generater)
            img.save(os.path.join(savedir, 'generated.png'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
