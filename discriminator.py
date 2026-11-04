import tensorflow as tf
from tensorflow.keras import layers, models

def build_rgb_discriminator(img_shape=(256, 256, 3), init_filters=64):
    model = models.Sequential(name="RGB_Discriminator")
    
    # Layer 1: 256x256x3 -> 128x128x64 (No BatchNorm on the input layer per DCGAN spec)
    model.add(layers.Input(shape=img_shape))
    model.add(layers.Conv2D(init_filters, kernel_size=4, strides=2, padding='same', use_bias=True))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    # Layer 2: 128x128x64 -> 64x64x128
    model.add(layers.Conv2D(init_filters * 2, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    # Layer 3: 64x64x128 -> 32x32x256
    model.add(layers.Conv2D(init_filters * 4, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    # Layer 4: 32x32x256 -> 16x16x512
    model.add(layers.Conv2D(init_filters * 8, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    # Layer 5: 16x16x512 -> 8x8x512
    model.add(layers.Conv2D(init_filters * 8, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    # Layer 6: 8x8x512 -> 4x4x1024
    model.add(layers.Conv2D(init_filters * 16, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    # Final Layer: Fully Convolutional map to single scalar logic 
    # Transforms (4, 4, 1024) spatial grid directly down to a (1, 1, 1) decision map
    model.add(layers.Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=True))
    model.add(layers.Flatten()) # Flattens a 1x1x1 matrix cleanly into a single scalar logit
    
    # Explicitly ensure final output is cast to float32 to prevent underflow in mixed precision
    model.add(layers.Activation('linear', dtype="float32")) 
    
    return model

if __name__ == "__main__":
    discriminator = build_rgb_discriminator()
    print(discriminator.summary())