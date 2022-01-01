import os
import random
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw
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
figure.savefig("out/_histogram.png", bbox_inches='tight')

noiseTypes = [
    IGN,
    blueNoise,
    whiteNoise,
    bayer
]

noiseTypeLabels = [
    "ign",
    "blue",
    "white",
    "bayer"
]

# Make larger images of the noises
for noise, label in zip(noiseTypes, noiseTypeLabels):
    im = Image.fromarray(np.uint8(noise*255.0)).resize((256,256), resample=PIL.Image.NEAREST)
    im.save("out/_big_"+label+".png")

regions = [
    (5,5),
    (10,7),
    (12,8),
    (2, 10)
]

regionColors=[
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#ffff00"
]


# Make numberlines of regions of noise images
for noise, label in zip(noiseTypes, noiseTypeLabels):
    im = Image.fromarray(np.uint8(noise*255.0)).resize((256,256), resample=PIL.Image.NEAREST).convert('RGB')
    im_e = ImageDraw.Draw(im)

    figure, axis = plt.subplots(len(regions))

    xmin = 0
    xmax = 1
    y = 1
    height = 1    
    
    for region, regionColor, regionIndex in zip(regions, regionColors, range(len(regions))):
        im_e.rectangle([(region[0]*16, region[1]*16), ((region[0]+3)*16, (region[1]+3)*16)], outline=regionColor, width=3)

        axis[regionIndex].set_xlim(0,1)
        axis[regionIndex].set_ylim(0,2)

        axis[regionIndex].hlines(y, xmin, xmax)
        axis[regionIndex].vlines(xmin, y - height / 2., y + height / 2.)
        axis[regionIndex].vlines(xmax, y - height / 2., y + height / 2.)
        
        for offset in range(0,9):
            offsetx = int(offset % 3)
            offsety = int(offset / 3)
            x = noise[region[0]+offsetx, region[1]+offsety]

            axis[regionIndex].plot(x, y, 'o', ms = 10, color=regionColor)

        axis[regionIndex].axis('off')

        axis[regionIndex].text(xmin - 0.01, y, '0', horizontalalignment='right')
        axis[regionIndex].text(xmax + 0.01, y, '1', horizontalalignment='left')       

    figure.tight_layout()
    figure.savefig("out/big_windows_" + label + "_" + str(regionIndex) + ".dft.png", bbox_inches='tight')

    im2 = Image.open("out/big_windows_" + label + "_" + str(regionIndex) + ".dft.png")

    imoutw = im.size[0] + im2.size[0]
    imouth = max(im.size[1], im2.size[1])
    imout = Image.new('RGB', (imoutw, imouth), (255, 255, 255))

    imout.paste(im, (0, int((imouth - im.size[1])/2)))
    imout.paste(im2, (im.size[0], int((imouth - im2.size[1])/2)))
    
    imout.save("out/_big_windows_"+label+".png")

