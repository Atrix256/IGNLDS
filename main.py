import os
import random
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import statistics

def R2Mask(x, y):
    # generalized golden ratio, from:
    # http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    g = 1.32471795724474602596;
    a1 = 1.0 / g;
    a2 = 1.0 / (g * g);
    return ((a1 * float(x)) + (a2 * float(y)))%1;

os.makedirs("out", exist_ok=True)
fnt = ImageFont.truetype("arial.ttf", 40)

# Make White Noise Texture
if False:
    whiteNoise = np.random.random(256)
    whiteNoise = whiteNoise.reshape((16, 16))
    im = Image.fromarray(np.uint8(whiteNoise*255.0))
    im.save("source/white16x16.png")

# Make IGN Texture
if False:
    IGN = np.empty(256)
    for i in range(256):
        x = i % 16
        y = int(i / 16)
        IGN[i] = (52.9829189 * ((0.06711056*float(x) + 0.00583715*float(y)) % 1)) % 1
    IGN = IGN.reshape((16, 16))
    im = Image.fromarray(np.uint8(IGN*255.0))
    im.save("source/ign16x16.png")

# Make Plus Texture
if False:
    PlusLDG = np.empty(256)
    for i in range(256):
        x = i % 16
        y = int(i / 16)
        PlusLDG[i] = (((float(x) + 3.0*float(y))/5) % 1)
    PlusLDG = PlusLDG.reshape((16, 16))
    im = Image.fromarray(np.uint8(PlusLDG*255.0))
    im.save("source/plus16x16.png")

# Make R2 Texture
if False:
    R2 = np.empty(256)
    for i in range(256):
        x = i % 16
        y = int(i / 16)
        R2[i] = R2Mask(x,y)
    R2 = R2.reshape((16, 16))
    im = Image.fromarray(np.uint8(R2*255.0))
    im.save("source/r216x16.png")

# Load noise
blueNoise = np.array(Image.open("source/bluenoise16x16.png")).astype(float) / 255.0
bayer = np.array(Image.open("source/bayer16x16.png"))[:,:,0].astype(float) / 255.0
whiteNoise = np.array(Image.open("source/white16x16.png")).astype(float) / 255.0
IGN = np.array(Image.open("source/ign16x16.png")).astype(float) / 255.0
R2 = np.array(Image.open("source/r216x16.png")).astype(float) / 255.0
PlusLDG = np.array(Image.open("source/plus16x16.png")).astype(float) / 255.0

# make combined noise images
imout = Image.new('RGB', (117, 22), (255, 255, 255))
imout.paste(Image.fromarray(np.uint8(IGN*255.0)), (3, 3))
imout.paste(Image.fromarray(np.uint8(whiteNoise*255.0)), (22, 3))
imout.paste(Image.fromarray(np.uint8(blueNoise*255.0)), (41, 3))
imout.paste(Image.fromarray(np.uint8(bayer*255.0)), (60, 3))
imout.paste(Image.fromarray(np.uint8(R2*255.0)), (79, 3))
imout.paste(Image.fromarray(np.uint8(PlusLDG*255.0)), (98, 3))
imout.save("out/_noises.png")

imout = imout.resize((1844,352), resample=PIL.Image.NEAREST)
imout_e = ImageDraw.Draw(imout)
imout_e.text((128,5), "IGN", font=fnt, fill=(0,0,0,255))
imout_e.text((432,5), "White", font=fnt, fill=(0,0,0,255))
imout_e.text((740,5), "Blue", font=fnt, fill=(0,0,0,255))
imout_e.text((1030,5), "Bayer", font=fnt, fill=(0,0,0,255))
imout_e.text((1320,5), "R2", font=fnt, fill=(0,0,0,255))
imout_e.text((1610,5), "Plus", font=fnt, fill=(0,0,0,255))
imout.save("out/_noisesBig.png")

# Make histograms
figure, axis = plt.subplots(2, 3)

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

axis[0,2].set_title("R2")
axis[0,2].set_xlabel("Value")
axis[0,2].set_ylabel("Count")
axis[0,2].hist(R2.reshape(256), 256, facecolor='blue', alpha=0.5)

axis[1,2].set_title("Plus")
axis[1,2].set_xlabel("Value")
axis[1,2].set_ylabel("Count")
axis[1,2].hist(PlusLDG.reshape(256), 256, facecolor='blue', alpha=0.5)

figure.tight_layout()
figure.savefig("out/_histogram.png", bbox_inches='tight')

noiseTypes = [
    IGN,
    whiteNoise,
    blueNoise,
    bayer,
    R2,
    PlusLDG
]

noiseTypeLabels = [
    "ign",
    "white",
    "blue",
    "bayer",
    "R2",
    "Plus"
]

# Make stochastic alpha test
imout = Image.new('RGB', (512+10, 768+15), (255, 255, 255))
for noise, label, noiseIndex in zip(noiseTypes, noiseTypeLabels, range(len(noiseTypes))):
    forest = np.array(Image.open("source/forest.png")).astype(float) / 255.0
    testsize = 16
    startx = int((forest.shape[0] - testsize)/2)
    starty = int((forest.shape[1] - testsize)/2)
    opacity = 1.0 / 9.0
    for ix in range(testsize):
        for iy in range(testsize):
            rng = noise[ix % 16, iy % 16]
            if rng < opacity:
                forest[startx + ix, starty + iy] = (1, 0, 1, 1)
    im = Image.fromarray(np.uint8(forest*255.0)).resize((256,256), resample=PIL.Image.NEAREST)
    im_e = ImageDraw.Draw(im)
    im_e.text((5,5), label, font=fnt, fill=(255,255,255,255))
    # im.save("out/_transparency_"+label+".png")
    imout.paste(im, ((noiseIndex%2)*266, int(noiseIndex/2)*266))

imout.save("out/_transparency.png")
            

# Make larger images of the noises
#for noise, label in zip(noiseTypes, noiseTypeLabels):
#    im = Image.fromarray(np.uint8(noise*255.0)).resize((256,256), resample=PIL.Image.NEAREST)
#    im.save("out/_big_"+label+".png")

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

        regionvalues = np.empty(9)
        
        for offset in range(0,9):
            offsetx = int(offset % 3)
            offsety = int(offset / 3)
            regionvalues[offset] = noise[region[0]+offsetx, region[1]+offsety]
            axis[regionIndex].plot(regionvalues[offset], y, 'o', ms = 10, color=regionColor)

        regionvalues = np.sort(regionvalues)
        distances = np.empty(9)
        for i in range(0, 9):
            distances[i] = (regionvalues[(i+1)%9] - regionvalues[i]) % 1

        axis[regionIndex].set_title("Distance std. dev. = " + "{:.3f}".format(statistics.stdev(distances)))

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

    imout_e = ImageDraw.Draw(imout)
    
    imout.save("out/_big_windows_"+label+".png")

