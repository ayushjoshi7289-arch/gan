import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def build_rgb_generator(latent_dim=100, init_filters=64):
    model = models.Sequential(name="RGB_Generator")
    model.add(layers.Input(shape=(latent_dim,)))
    
    # Start: Take latent vector and project/reshape into a small, deep 3D block
    # 4 * 4 * 1024 = 16,384 neurons
    model.add(layers.Dense(4 * 4 * init_filters * 16, use_bias=False))
    model.add(layers.Reshape((4, 4, init_filters * 16)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU()) # Generators traditionally use standard ReLU
    
    # Layer 1: Upsample 4x4 -> 8x8
    model.add(layers.Conv2DTranspose(init_filters * 8, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Layer 2: Upsample 8x8 -> 16x16
    model.add(layers.Conv2DTranspose(init_filters * 8, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Layer 3: Upsample 16x16 -> 32x32
    model.add(layers.Conv2DTranspose(init_filters * 4, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Layer 4: Upsample 32x32 -> 64x64
    model.add(layers.Conv2DTranspose(init_filters * 2, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Layer 5: Upsample 64x64 -> 128x128
    model.add(layers.Conv2DTranspose(init_filters, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Layer 6: Upsample 128x128 -> 256x256x3 (Final RGB Output)
    model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same'))
    # Tanh activation squashes output pixels precisely between -1 and 1
    model.add(layers.Activation('tanh', dtype="float32")) # Ensure output is float32 for visualization and loss calculations

    
    
    return model

if __name__ == "__main__":
    # Example usage: Build and summarize the generator and discriminator
    latent_dim = 100
    generator = build_rgb_generator(latent_dim=latent_dim)

    
    print(generator.summary())
