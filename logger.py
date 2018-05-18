# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import numpy as np
import nsml
import visdom
from torchvision.utils import save_image

from utils import *

class Logger(object):
    def __init__(self, log_dir=None):
        self.last = None
        self.log_dir = log_dir
        if nsml.IS_ON_NSML:
            self.viz = nsml.Visdom(visdom=visdom)

    def scalar_summary(self, tag, value, step, scope=None):
        if nsml.IS_ON_NSML:
            if self.last and self.last['step'] != step:
                nsml.report(**self.last,scope=scope)
                self.last = None
            if self.last is None:
                self.last = {'step':step,'iter':step,'epoch':1}
            self.last[tag] = value

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        if nsml.IS_ON_NSML:
            self.viz.images(
                to_np(images),
                opts=dict(title='%s/%d' % (tag, step), caption='%s/%d' % (tag, step)),
            )
        else:
            save_image(
                images.data,
                '{}/step{}.png'.format(self.log_dir, step),
                nrow=8,
                normalize=True
            )
