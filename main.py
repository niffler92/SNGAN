import os
import argparse
from io import BytesIO
import base64

import numpy as np
import nsml
from PIL import Image

from dataloader import get_loader
from trainer import Trainer
from discriminator import Discriminator
from generator import Generator
from logger import Logger


def main(args, scope):
    train_loader, _ = get_loader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    G = Generator(args)
    D = Discriminator(args)
    trainer = Trainer(train_loader, G, D, args)

    save, load, infer = get_bindings(trainer)
    nsml.bind(save=save, load=load, infer=infer)

    if args.pause:
        nsml.paused(scope=scope)

    if args.mode == 'train':
        if args.verbose:
            trainer.show_current_model()
        trainer.train()
    elif args.mode == 'sample':
        trainer.sample()


def get_bindings(trainer):
    def save(filename, *args):
        trainer.save(filename)

    def load(filename, *args):
        trainer.load(filename)

    def infer(input):
        result = trainer.infer(input)
        # convert tensor to dataurl
        data_url_list = [''] * input
        for idx, sample in enumerate(result):
            numpy_array = np.uint8(sample.cpu().numpy()*255)
            image = Image.fromarray(np.transpose(numpy_array, axes=(1, 2, 0)), 'RGB')
            temp_out = BytesIO()
            image.save(temp_out, format='png')
            byte_data = temp_out.getvalue()
            data_url_list[idx] = u'data:image/{format};base64,{data}'.\
                format(format='png',
                       data=base64.b64encode(byte_data).decode('ascii'))
        return data_url_list

    return save, load, infer


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SN-GAN")

    # Dataset
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        choices=['CIFAR10'])
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Model params
    parser.add_argument("--sn", action='store_true', help="Boolean to conduct spectral normalization")
    parser.add_argument("--z_dim", default=128, type=int,
                        help="Dimension of latent vector")

    # Training settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=int, default=2e-4)
    parser.add_argument('--g_iter', type=int, default=1)
    parser.add_argument('--d_iter', type=int, default=5)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'])
    parser.add_argument('--log_path', type=str, default='./cifar10/logs')
    parser.add_argument('--model_save_path', type=str, default='./cifar10/models')
    parser.add_argument('--nsamples', type=int, default=64)
    parser.add_argument('--inception_score', action='store_true')

    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--verbose', action='store_true')

    # nsml setting
    parser.add_argument('--pause', type=int, default=0)

    args = parser.parse_args()
    if not os.path.exists(args.dataset.lower()):
        os.makedirs(args.dataset.lower())
    if args.dataset == "CIFAR10":
        args.m_g = 4
        args.ngf = 512
        args.ndf = 512
    main(args, scope=locals())
