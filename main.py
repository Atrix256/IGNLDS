import os
import random
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt

os.makedirs("out", exist_ok=True)

# Make White Noise Texture
whiteNoise = np.random.random(256)
whiteNoise = whiteNoise.reshape((16, 16))
im = Image.fromarray(np.uint8(whiteNoise*255.0))
im.save("out/white16x16.png")

# Make IGN Texture
IGN = np.empty(256)
for i in range(256):
    x = i % 16
    y = int(i / 16)
    IGN[i] = (52.9829189 * ((0.06711056*float(x) + 0.00583715*float(y)) % 1)) % 1
IGN = IGN.reshape((16, 16))
im = Image.fromarray(np.uint8(IGN*255.0))
im.save("out/ign16x16.png")

# Load blue noise
blueNoise = np.array(Image.open("source/bluenoise16x16.png")).astype(float) / 255.0

# Load bayer
bayer = np.array(Image.open("source/bayer16x16.png"))[:,:,0].astype(float) / 255.0


# Make histograms
figure, axis = plt.subplots(2, 2)

figure.suptitle("16x16 Texture Histograms, 256 buckets")

axis[0,0].set_title("IGN")
axis[0,0].set_xlabel("Value")
axis[0,0].set_ylabel("Count")
axis[0,0].hist(IGN.reshape(256), 256, facecolor='blue', alpha=0.5)

axis[1,0].set_title("Blue Noise")
axis[1,0].set_xlabel("Value")
axis[1,0].set_ylabel("Count")
axis[1,0].hist(blueNoise.reshape(256), 256, facecolor='blue', alpha=0.5)

axis[0,1].set_title("White Noise")
axis[0,1].set_xlabel("Value")
axis[0,1].set_ylabel("Count")
axis[0,1].hist(whiteNoise.reshape(256), 256, facecolor='blue', alpha=0.5)

axis[1,1].set_title("Bayer")
axis[1,1].set_xlabel("Value")
axis[1,1].set_ylabel("Count")
axis[1,1].hist(bayer.reshape(256), 256, facecolor='blue', alpha=0.5)

figure.tight_layout()
figure.savefig("out/_histogram.png")#, bbox_inches='tight')

noiseTypes = [
    IGN,
    blueNoise,
    whiteNoise,
    bayer
]

noiseTypeLabels = [
    "IGN",
    "Blue Noise",
    "White Noise",
    "Bayer"
]

# Make larger images of the noises
for noise, label in zip(noiseTypes, noiseTypeLabels):
    im = Image.fromarray(np.uint8(noise*255.0)).resize((256,256), resample=PIL.Image.NEAREST)
    im.save("out/_"+label+"_big.png")
