from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Reshape, Flatten 
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam 

import matplotlib.pyplot as plt 
import numpy as np 

img_rows = 28
img_cols = 28
channels = 3 
img_shape = (img_rows, img_cols, channels)

z_dim = 100 

"""
    Generator 
"""
def generator(img_shape, z_dim):
    model = Sequential() 

    # Hidden Layer 
    model.add(Dense(128, input_dim=z_dim))

    # LeakyReLU 
    model.add(LeakyReLU(alpha=0.1))

    # Output layer with tanh activation 
    model.add(Dense(img_cols * img_rows * channels, activation='tanh'))
    model.add(Reshape(img_shape))

    z = Input(shape=(z_dim,))
    img = model(z)

    return Model(z, img)


"""
    Discriminator 
"""
def discriminator(img_shape):
    model = Sequential() 

    model.add(Flatten(input_shape=img_shape))

    # Hidden Layer 
    model.add(Dense(128))

    # Leaky ReLU 
    model.add(LeakyReLU(alpha=0.01))
    # Output layer with sigmoid activation 
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    prediction = model(img)

    return Model(img, prediction)


# Build and compile the Discriminator 
discriminator = discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Build the Generator 
generator = generator(img_shape, z_dim)

# Generated image to be sed as input 
z = Input(shape=(100,))
img = generator(z)

# Keep Discriminator's parameters constant during Generator training 
discriminator.trainable = False 

# The discriminator's prediction 
prediction = discriminator(img)

# Combined GAN model to train the Generator 
combined = Model(z, prediction)
combined.compile(loss='binary_crossentropy', optimizer=Adam())


"""
    Training 
"""
losses = [] 
accuracies = [] 

def read_images():
    imgs = [] 
    for i in range(1, 100000 + 1):
        filename = str(i).zfill(6) + ".jpg" 
        filepath = "../../../../data/img_align_celeba/" + filename
        img = load_img(filepath)
        x = img_to_array(img)
        x.resize(img_rows, img_cols, channels)
        x[:, :, 0] = x[:, :, 0] / 127.5 - 1.
        x[:, :, 1] = x[:, :, 1] / 127.5 - 1.
        x[:, :, 2] = x[:, :, 2] / 127.5 - 1.
        
        imgs.append(x)
        if (i % 1000 == 0):
            print('%d images readed' % i)
    return np.asarray(imgs) 

def train(iterations, batch_size, sample_interval):
    # Load the dataset 
    X_train = read_images()
    print(X_train.shape)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        # -------------------------------------
        # Train the Discriminator 
        # -------------------------------------

        # Select a random batch of real images 
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generate a batch of fake images 
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Discriminator loss 
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ------------------------------------
        # Train the Generator
        # ------------------------------------

        # Generate a batch of fake images 
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Generator loss 
        g_loss = combined.train_on_batch(z, real)

        if iteration % sample_interval == 0:

            # Output training progress 
            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (iteration, d_loss[0], 100 * d_loss[1], g_loss))

            # Save losses and accuracies to be plotted after training 
            losses.append((d_loss[0], g_loss))
            accuracies.append(100 * d_loss[1])

            # Output generated image samples 
            sample_images(iteration)

def sample_images(iteration, image_grid_rows=4, image_grid_columns=4):
    
    # Sample random noise 
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise 
    gen_imgs = generator.predict(z)

    # Rescale images to 0 - 1 
    gen_imgs = 0.5 * gen_imgs + 0.5 

    # set image grid
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)

    cnt = 0 
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output image grid 
            axs[i, j].imshow(gen_imgs[cnt, :, :, :], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig('../../../../output/celeba_%d.png' % iteration)
    

    


import warnings; warnings.simplefilter('ignore')

iterations = 200000
batch_size = 128 
sample_interval = 1000 

# Train the GAN for the specified number of iterations 
train(iterations, batch_size, sample_interval)



