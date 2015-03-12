from skimage.transform import warp, AffineTransform as af
import numpy as np
import random
import math


def random_transform(im, threshold=0.5):
    x, y = im.shape
    pixel_count = x * y
    
    rotation = random.uniform(0, 360)
    flip = np.random.binomial(1, 0.5)
    if flip:
        im = np.fliplr(im)
    rotation = random.uniform(0, 360)
    translation = random.uniform(4.0/float(x),
                                 4.0/float(y))
    zoom = math.exp(random.uniform(math.log(1.0/1.0), math.log(1.1)))
    transform = af(
        scale=(zoom, zoom),
        rotation=rotation,
        translation=translation
    )
    transformed_im = warp(im, transform, cval=1.0)
    orig_white_pixels =\
        np.bincount(np.unique(im, return_inverse=True)[1])[-1]
    orig_signal_pct =\
        float(pixel_count - orig_white_pixels)/float(pixel_count)
    transformed_white_pixels =\
        np.bincount(
            np.unique(transformed_im,
                      return_inverse=True)[1])[-1]
    transformed_signal_pct =\
        float(pixel_count - transformed_white_pixels)/float(pixel_count)
    pix_pct_diff = 1.0 - (orig_signal_pct-transformed_signal_pct)/orig_signal_pct
    if pix_pct_diff >= threshold:
        return transformed_im
    return random_transform(im)


def random_transform_batch(batch):
    x, y, batch_size = batch.shape
    output_batch = np.zeros(batch.shape)
    for i in xrange(batch_size):
        output_batch[:, :, i] = random_transform(batch[:, :, i], threshold)
    return output_batch

def random_transform_block(inputs):
    batch = np.array([random_transform_batch(b) for b in inputs])
    return batch[..., np.newaxis]