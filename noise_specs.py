import numpy as np, argparse, h5py, os, sys
from enlib import fft, utils, enmap, errors, config
from enact import filedb, data
from mpi4py import MPI
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("--nx", type=int, default=1000)
parser.add_argument("--ny", type=int, default=600)
parser.add_argument("--sx", type=str, default="lin")
parser.add_argument("--sy", type=str, default="log")
parser.add_argument("-x", "--xrange", type=str, default="0:200")
parser.add_argument("-y", "--yrange", type=str, default=None)
parser.add_argument("-N", "--apply-nmat", action="store_true")
parser.add_argument("-c", action="store_true")
args = parser.parse_args()

yrange = args.yrange or ("1e-8:1e-3" if args.apply_nmat else "1e1:1e6")

comm = MPI.COMM_WORLD
utils.mkdir(args.odir)
srate = 400

class Axis:
	def __init__(self, n, scale, range):
		self.n, self.range, self.scale = n, range, scale
		self.f = np.log10 if scale == "log" else lambda x:x
		self.fr = self.f(range)
	def __call__(self, v):
		return np.minimum(self.n-1,np.floor((self.f(v)-self.fr[0])/(self.fr[1]-self.fr[0])*self.n).astype(np.int32))

def parse_axis(n, scale, range): return Axis(n, scale, [float(w) for w in range.split(":")])

xaxis = parse_axis(args.nx, args.sx, args.xrange)
yaxis = parse_axis(args.ny, args.sy, yrange)
xinds = np.arange(xaxis.n)

shape, wcs = enmap.geometry(pos=np.array([yaxis.fr,xaxis.fr]).T*np.pi/180, shape=(yaxis.n,xaxis.n), proj="car")

filedb.init()
ids = filedb.scans[args.sel].ids
for si in range(comm.rank, len(ids), comm.size):
	id    = ids[si]
	entry = filedb.data[id]
	ofile = "%s/%s.fits" % (args.odir, id)
	if args.c and os.path.isfile(ofile): continue
	print "reading %s" % id
	try:
		d     = data.read(entry)
		d     = data.calibrate(d)
	except (IOError, errors.DataMissing) as e:
		print "skipping (%s)" % e.message
		continue
	if args.apply_nmat:
		d.tod = d.noise.apply(d.tod)
	ft    = fft.rfft(d.tod)
	ps    = np.abs(ft)**2/(d.tod.shape[1]*srate)
	freq  = np.linspace(0,200,ps.shape[1])

	canvas = enmap.zeros(shape, wcs,dtype=np.int32)
	x = xaxis(freq)
	for ps_det in ps:
		ps_bin = np.bincount(x, ps_det, minlength=xaxis.n)/np.bincount(x, minlength=xaxis.n)
		D_bin = ps_bin**0.5
		y  = yaxis(D_bin)
		y  = np.maximum(0,np.minimum(canvas.shape[0]-1,y))
		canvas[(y, xinds)] += 1

	enmap.write_map(ofile, canvas)
