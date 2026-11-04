import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_history.csv")

plt.figure(figsize=(10,5))

plt.plot(df["epoch"], df["gen_loss"],
         label="Generator Loss")

plt.plot(df["epoch"], df["disc_loss"],
         label="Discriminator Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training History")
plt.legend()
plt.grid(True)

plt.show()