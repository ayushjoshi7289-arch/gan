import tensorflow as tf

# Setup Cross Entropy Loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 1. Initialize Native Optimizers
# Under the global mixed_float16 policy, standard Keras optimizers manage 
# loss scaling natively. Manual wrappers are omitted to avoid conflict.
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)


@tf.function(jit_compile=True)
def train_discriminator_step(real_images, generator, discriminator, latent_dim=100):
    """
    Trains the discriminator for one step.
    Explicitly passing generator and discriminator avoids graph tracing issues.
    """
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, latent_dim])

    # Generate fake images outside the tape and stop gradients
    fake_images = generator(noise, training=True)
    fake_images = tf.stop_gradient(fake_images)

    with tf.GradientTape() as disc_tape:
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        # Label smoothing applied to real images (0.9 instead of 1.0)
        disc_loss_real = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
        disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

    # Native mixed precision automatically scales, computes, and applies gradients safely
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    return disc_loss


@tf.function(jit_compile=True)
def train_generator_step(batch_size, generator, discriminator, latent_dim=100):
    """
    Trains the generator for one step.
    """
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)

        # Generator wants discriminator to think fake images are real (label = 1.0)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    # Native mixed precision automatically scales, computes, and applies gradients safely
    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return gen_loss