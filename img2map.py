import numpy as np, PIL.Image, argparse
from enlib import enmap, wcsutils
parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("template", nargs="?")
parser.add_argument("ofile")
parser.add_argument("-m", "--mask", action="store_true")
args = parser.parse_args()

# Read image into [{r,g,b},ny,nx]
img = np.rollaxis(np.array(PIL.Image.open(args.image)),2)
# Create output map based on template
if args.template:
	template = enmap.read_map(args.template)
	assert img.shape[-2:] == template.shape[-2:], "Image and template shapes do not conform"
	wcs = template.wcs
else:
	wcs = wcsutils.WCS(naxis=2)
res = enmap.zeros(img.shape, wcs, dtype=np.int16)
# Copy over data, taking into account y ordering
res[:] = img[:,::-1,:]
if args.mask: res = np.any(res,0).astype(np.int16)
# And write
enmap.write_map(args.ofile, res)
