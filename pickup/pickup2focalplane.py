import numpy as np, argparse
from scipy import ndimage
from enlib import enmap, utils
from enact import files
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+", help="imap imap ... layout layout ... if not transpose, else interlaced")
parser.add_argument("ofile")
parser.add_argument("-s", "--step", type=int,   default=3)
parser.add_argument("-r", "--res",  type=float, default=0.2)
parser.add_argument("-R", "--rad",  type=float, default=1.0)
parser.add_argument("-T", "--transpose", action="store_true")
parser.add_argument("-i", "--individual", action="store_true")
parser.add_argument("--slice", type=str, default="")
args = parser.parse_args()

dtype = np.float32
ncol, nrow = 32, 33
nfile = len(args.ifiles)/2
rad   = args.rad * utils.arcmin * utils.fwhm
if not args.transpose:
	imapfiles = args.ifiles[:nfile]
	ilayfiles = args.ifiles[nfile:]
else:
	imapfiles = args.ifiles[0::2]
	ilayfiles = args.ifiles[1::2]

# Read in our focalplane layouts so we can define our output map bounds
dets, offs, boxes, imaps = [], [], [], []
for i in range(nfile):
	det, off = files.read_point_template(ilayfiles[i])
	imap = enmap.read_map(imapfiles[i])
	if args.slice: imap = eval("imap"+args.slice)
	# We want y,x-ordering
	off = off[:,::-1]
	box = utils.minmax(off,0)
	dets.append(det)
	offs.append(off)
	boxes.append(box)
	imaps.append(imap)
box = utils.bounding_box(boxes)
box = utils.widen_box(box, rad*5, relative=False)

# We assume that the two maps have the same pixelization
imaps = enmap.samewcs(np.array(imaps), imaps[0])
# Downsample by averaging
imaps = enmap.downgrade(imaps, (1,args.step))
naz   = imaps.shape[-1]

# Ok, build our output geometry
shape, wcs = enmap.geometry(pos=box, res=args.res*utils.arcmin, proj="car", pre=(naz,))
omap = enmap.zeros(shape, wcs, dtype=dtype)

# Normalization
norm = enmap.zeros(shape[-2:],wcs)
norm[0,0] = 1
norm = enmap.smooth_gauss(norm, rad)[0,0]

# Loop through slices and populate
bazs = []
for iaz in range(naz):
	# Get our boresight az
	bazs.append(imaps.pix2sky([0,iaz])[1])
	vals = []
	for i in range(nfile):
		# Go from detectors to y-pixel in input maps
		ypix = utils.transpose_inds(dets[i], nrow, ncol)
		vals.append(imaps[i,ypix,iaz])
	vals = np.concatenate(vals)
	pos  = np.concatenate(offs)
	# Write to appropriate position in array
	pix  = np.maximum(0,np.minimum((np.array(shape[-2:])-1)[:,None],enmap.sky2pix(shape, wcs, pos.T).astype(np.int32)))
	m    = enmap.zeros(shape[-2:],wcs)
	m[tuple(pix)] = vals
	# Grow by smoothing
	m = enmap.smooth_gauss(m, rad)/norm
	omap[iaz] = m

# Truncate very low values
refval = np.mean(np.abs(omap))*0.01
mask = np.any(np.abs(omap)>refval,0)
omap[:,~mask] = 0

if not args.individual:
	# output a single file
	enmap.write_map(args.ofile, omap)
else:
	# output one file for each. This lets us encode
	# each one's azimuth
	utils.mkdir(args.ofile)
	for i, (baz, m) in enumerate(zip(bazs, omap)):
		if np.all(m==0): continue
		print i
		m.wcs.wcs.crval[0] = baz/utils.degree
		enmap.write_map(args.ofile + "/map%04d.fits" % i, m)
