import numpy as np
from PIL import Image

def preprocess_input(x):
    # [0,255] => [-1,1]
    x = (x - 127.5) / 127.5
    return x

def decode_output(x):
    # [-1,1] => [0,255]
    x = x * 127.5 + 127.5
    return x

def create_generated_img(generator, num_generate_imgs=144):

    # generate images
    z_dim = generator.input_shape[-1]
    z = np.random.uniform(-1,1,size=(num_generate_imgs, z_dim))
    gen = decode_output(generator.predict_on_batch(z))
    gen = np.clip(gen, 0., 255.).astype(np.uint8)

    # for single-channel images
    if (gen.shape[-1] == 1):
        gen = gen.reshape(gen.shape[0], gen.shape[1], gen.shape[2])

    # concatenate images
    grid_size = int(np.sqrt(num_generate_imgs))
    rows = []
    for i in range(0, num_generate_imgs, grid_size):
        row = np.concatenate(gen[i:i+grid_size], axis=1)
        rows.append(row)
    final = np.concatenate(rows, axis=0)
    return Image.fromarray(final)
