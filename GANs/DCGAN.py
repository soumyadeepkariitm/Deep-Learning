import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from glob import glob
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_H = 64
IMG_W = 64
IMG_C = 3
latent_dim = 128
w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img

def tf_dataset(images_path, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

discriminator = keras.Sequential([
    InputLayer(input_shape = (IMG_H, IMG_W, IMG_C)),
    Conv2D(64, 5, 2, padding = "same",kernel_initializer=w_init),
    LeakyReLU(0.2),
    Dropout(0.3),
    Conv2D(128, 5, 2, padding = "same",kernel_initializer=w_init),
    LeakyReLU(0.2),
    Dropout(0.3),
    Conv2D(256, 5, 2, padding = "same",kernel_initializer=w_init),
    LeakyReLU(0.2),
    Dropout(0.3),
    Conv2D(512, 5, 2, padding = "same",kernel_initializer=w_init),
    LeakyReLU(0.2),
    Dropout(0.3),
    Flatten(),
    Dense(1)
    ], name = "discriminator"
    )
discriminator.summary()
generator = keras.Sequential([
    InputLayer(input_shape = (latent_dim,)),
    Dense(512 * 4 *4, use_bias=False),
    BatchNormalization(),
    LeakyReLU(0.2),
    Reshape(target_shape = (4, 4, 512)),
    Conv2DTranspose(256, 5, 2, padding = 'same', use_bias=False,kernel_initializer=w_init),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2DTranspose(128, 5, 2, padding = 'same', use_bias=False,kernel_initializer=w_init),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2DTranspose(64, 5, 2, padding = 'same', use_bias=False,kernel_initializer=w_init),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2DTranspose(32, 5, 2, padding = 'same', use_bias=False,kernel_initializer=w_init),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2D(3, 5, 1, padding = "same", activation = 'tanh')
    ], name = "generator"
    )
generator.summary()

class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]


            ## Train the discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        generated_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as ftape:
            predictions = self.discriminator(generated_images)
            d1_loss = self.loss_fn(generated_labels, predictions)
        grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            ## Train the discriminator
        labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as rtape:
            predictions = self.discriminator(real_images)
            d2_loss = self.loss_fn(labels, predictions)
        grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gtape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss}
    
def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])  
    filename = f"samples/generated_plot_epoch-{epoch+1}.png" 
    pyplot.savefig(filename)
    pyplot.close()


if __name__ == "__main__":
    ## Hyperparameters
    batch_size = 128
    num_epochs = 80
    images_path = glob("archive/images/*")
    
    #discriminator.load_weights("saved_model/d_model.h5")
    #generator.load_weights("saved_model/g_model.h5")
    
    gan = GAN(discriminator, generator, latent_dim)
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1) # label smoothing to lower the confidence
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)

    images_dataset = tf_dataset(images_path, batch_size)

    for epoch in range(num_epochs):
        gan.fit(images_dataset, epochs=1)
        generator.save("saved_model/g_model.h5")
        discriminator.save("saved_model/d_model.h5")

        n_samples = 25
        noise = np.random.normal(size=(n_samples, latent_dim))
        examples = generator.predict(noise)
        save_plot(examples, epoch, int(np.sqrt(n_samples)))
