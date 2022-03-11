import argparse
parser = argparse.ArgumentParser()
parser.add_argument("colorbar_img")
args = parser.parse_args()
from PIL import Image
import numpy as np

with Image.open(args.colorbar_img) as img:
	data = np.array(img)[0]

desc = ""
for i, pix in enumerate(data):
	desc += "%.5f:%02x%02x%02x" % (i/(len(data)-1), pix[0], pix[1], pix[2])
	if i < len(data)-1: desc+=","
print(desc)
