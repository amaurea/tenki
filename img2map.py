import numpy as np, PIL.Image, argparse
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("template")
parser.add_argument("ofile")
args = parser.parse_args()

# Read image into [{r,g,b},ny,nx]
img = np.rollaxis(np.array(PIL.Image.open(args.image)),2)
# Create output map based on template
template = enmap.read_map(args.template)
assert img.shape[-2:] == template.shape[-2:], "Image and template shapes do not conform"
res = enmap.zeros(img.shape, template.wcs, dtype=np.int16)
# Copy over data, taking into account y ordering
res[:] = img[:,::-1,:]
# And write
enmap.write_map(args.ofile, res)
