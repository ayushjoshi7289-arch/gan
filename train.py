import glob
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# Enable mixed precision for modern GPU tensor core utilization
mixed_precision.set_global_policy("mixed_float16")

from constructor import build_rgb_generator
from discriminator import build_rgb_discriminator
from lossopt import train_discriminator_step, train_generator_step

plt.ion()  # Enable interactive mode for real-time plotting

# --- HARDWARE OPTIMIZATION ---
print("GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled dynamically.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU execution stack.")

# --- CONFIGURATION HYPERPARAMETERS ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 3
BATCH_SIZE = 16
LATENT_DIM = 100
EPOCHS = 50
DATA_DIR = r"archive/images"  # Path to your custom image folder
LOG_FILE = "training_history.csv"
LOG_COLUMNS = ["epoch", "gen_loss", "disc_loss", "execution_time_seconds"]
fixed_noise = tf.random.normal([16, LATENT_DIM])

# --- 1. DATA PREPROCESSING FUNCTIONS ---
def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0  # Normalized perfectly to [-1, 1] for Tanh
    return img

def load_dataset(data_dir, batch_size):
    list_ds = tf.data.Dataset.list_files(
        [
            os.path.join(data_dir, "*.jpg"),
            os.path.join(data_dir, "*.jpeg"),
            os.path.join(data_dir, "*.png")
        ],
        shuffle=True
    )
    ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(5000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- 2. MAIN EXECUTION PIPELINE ---
def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Target directory '{DATA_DIR}' not found. Please populate it first.")
        return
    
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob.glob(os.path.join(DATA_DIR, ext)))

    if len(image_files) == 0:
        raise ValueError(f"No usable images found in target location '{DATA_DIR}'.")

    print(f"Found {len(image_files)} source images.")
    
    if len(image_files) < BATCH_SIZE:
        raise ValueError(
            f"Found only {len(image_files)} images, but BATCH_SIZE={BATCH_SIZE}. "
            "Reduce your batch size configuration or add more data."
        )

    # Initialize model architectures
    print("Initializing Generative Adversarial Networks...")
    generator = build_rgb_generator(latent_dim=LATENT_DIM)
    discriminator = build_rgb_discriminator()

    # Build input execution pipeline
    train_ds = load_dataset(DATA_DIR, BATCH_SIZE)
    print("Dataset streaming pipeline built successfully.")

    cardinality = tf.data.experimental.cardinality(train_ds)
    if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
        print("Dataset cardinality cannot be structuralized upfront.")
    elif cardinality.numpy() == 0:
        raise ValueError(f"No validation data found in targeted path: '{DATA_DIR}'.")

    print(f"Dataset compiled into {cardinality.numpy()} total batches.")

    # Tracking metrics initialization
    history_data = []

    # Safe log setup
    if os.path.exists(LOG_FILE):
        print(f"Existing file found. Appending metric logs directly to '{LOG_FILE}'.")
    else:
        print(f"No existing profile detected. Creating a fresh tracking workspace at '{LOG_FILE}'.")
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_FILE, index=False)

    print(f"Beginning custom training process for {EPOCHS} epochs...\n")
    
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        epoch_gen_losses = []
        epoch_disc_losses = []
        
        for image_batch in train_ds:
            current_batch_size = image_batch.shape[0]
            
            # Step A: Optimize Discriminator (1 iteration)
            disc_loss = train_discriminator_step(image_batch, generator, discriminator, LATENT_DIM)
            epoch_disc_losses.append(float(disc_loss))

            # Step B: Optimize Generator (2 iterations intentional cadence)
            batch_gen_loss_running_total = 0.0
            for _ in range(2):
                gen_loss = train_generator_step(current_batch_size, generator, discriminator, LATENT_DIM)
                batch_gen_loss_running_total += float(gen_loss)

            # Keep average batch loss clean
            epoch_gen_losses.append(batch_gen_loss_running_total / 2.0)

        if not epoch_gen_losses:
            raise RuntimeError("Data collection failed. Stream produced empty sets.")
            
        # Fast numerical operations via local NumPy arrays
        avg_gen_loss = np.mean(epoch_gen_losses)
        avg_disc_loss = np.mean(epoch_disc_losses)
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Gen Loss: {avg_gen_loss:.4f} | Disc Loss: {avg_disc_loss:.4f} | Time: {elapsed_time:.1f}s")
        
        # Periodic visualization display tracking
        if epoch % 5 == 0:
            show_generated_images(generator, epoch)
        
        epoch_record = {
            "epoch": epoch,
            "gen_loss": avg_gen_loss,
            "disc_loss": avg_disc_loss,
            "execution_time_seconds": elapsed_time
        }
        history_data.append(epoch_record)
        
        # Guarded multi-epoch serialization blocks
        if epoch % 5 == 0 or epoch == EPOCHS:
            # Enforce deterministic column sequence before appending to file
            df = pd.DataFrame(history_data)[LOG_COLUMNS]
            df.to_csv(LOG_FILE, mode='a', index=False, header=False)
            history_data.clear() 
            
            generator.save_weights(f"generator_epoch_{epoch}.weights.h5")
            print(f" Saved checkpoint tracking parameters securely to local storage structure.")

    print(f"\nTraining pipeline concluded successfully. Final metrics verified in '{LOG_FILE}'.")


def show_generated_images(generator, epoch, n=4):
    images = generator(fixed_noise, training=False)
    images = (images + 1) / 2.0  # Scale backward cleanly from [-1, 1] to [0, 1]
    images = tf.clip_by_value(images, 0, 1)

    fig, axes = plt.subplots(n, n, figsize=(6, 6))
    idx = 0
    images = images.numpy()
    for i in range(n):
        for j in range(n):
            axes[i, j].imshow(images[idx])
            axes[i, j].axis("off")
            idx += 1
            
    os.makedirs("generated_samples", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"generated_samples/epoch_{epoch:03d}.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()