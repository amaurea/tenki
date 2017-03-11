import numpy as np, os, time, h5py
from scipy import optimize
from enlib import utils, config, enplot, mpi, errors, fft
from enlib import pmat, coordinates, enmap, bench, bunch
from enact import filedb, actdata, actscan
parser = config.ArgumentParser(os.environ["HOME"] + "./enkirc")
parser.add_argument("srclist")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("--rcol", type=int, default=6)
parser.add_argument("--dcol", type=int, default=7)
parser.add_argument("-f", "--fknee", type=float, default=10.0)
parser.add_argument("-a", "--alpha", type=float, default=10)
parser.add_argument("-R", "--radius",type=float, default=10)
parser.add_argument("-r", "--res",   type=float, default=0.1)
args = parser.parse_args()

config.default("pmat_accuracy", 10.0, "Factor by which to lower accuracy requirement in pointing interpolation. 1.0 corresponds to 1e-3 pixels and 0.1 arc minute in polangle")

filedb.init()
ids  = filedb.scans[args.sel]
comm = mpi.COMM_WORLD
R    = args.radius*utils.arcmin
res  = args.res*utils.arcmin
dtype= np.float64
utils.mkdir(args.odir)

def find_scan_vel(scan, ipos, aspeed, dt=0.1):
	hpos  = coordinates.transform("equ","hor", ipos, time=scan.mjd0, site=scan.site)
	hpos[0] += aspeed*dt
	opos  = coordinates.transform("hor","equ", hpos, time=scan.mjd0, site=scan.site)
	opos[0] = utils.rewind(opos[0],ipos[0])
	return (opos-ipos)/dt

def write_sdata(ofile, sdata):
	# Output thumb for this tod
	with h5py.File(ofile, "w") as hfile:
		for i, sdat in enumerate(sdata):
			g = hfile.create_group("%d"%i)
			g["map"]    = sdat.map
			g["div"]    = sdat.div
			g["sid"]    = sdat.sid
			g["id"]     = sdat.id
			g["srcpos"] = sdat.srcpos
			g["vel"]    = sdat.vel
			g["fknee"]  = sdat.fknee
			g["alpha"]  = sdat.alpha
			header = sdat.map.wcs.to_header()
			for key in header:
				g["wcs/"+key] = header[key]

# Load source database
srcpos = np.loadtxt(args.srclist, usecols=(args.rcol, args.dcol)).T*utils.degree
for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	# Check if we hit any of the sources
	poly      = filedb.scans.select([id]).data["bounds"][:,:,0]*utils.degree
	srcpos    = srcpos.copy()
	srcpos[0] = utils.rewind(srcpos[0], poly[0,0])
	sids      = np.where(utils.point_in_polygon(srcpos.T, poly.T))[0]
	if len(sids) == 0:
		print "%s has 0 srcs: skipping" % id
		continue
	print "%s has %d srcs: %s" % (id,len(sids),",".join([str(i) for i in sids]))
	entry = filedb.data[id]
	try:
		d = actdata.read(entry)
		d = actdata.calibrate(d, exclude=["autocut"])
	except errors.DataMissing as e:
		print "%s skipped: %s" % (id, e.message)
		continue
	tod = d.tod
	del d.tod
	# Apply high-pass filter. Will assume white tod after this
	with bench.show("filter"):
		freqs = fft.rfftfreq(d.nsamp)*d.srate
		ft    = fft.rfft(tod)
		ft[:,0]   = 0
		ft[:,1:] /= 1 + (freqs[1:]/args.fknee)**-args.alpha
		fft.ifft(ft, tod, normalize=True)
		tod = tod.astype(dtype)

	# Estimate white noise level, and weight tod by it
	ivar = 1/np.mean(tod**2,-1)
	tod *= ivar[:,None]

	# Find azimuth scanning speed
	aspeed = np.median(np.abs(d.boresight[1,1:]-d.boresight[1,:-1])[::10])*d.srate

	# Build a small, high-res map around each source
	sdata = []
	with bench.show("scan"):
		scan = actscan.ACTScan(entry, d=d)
	pcut = pmat.PmatCut(scan)
	junk = np.zeros(pcut.njunk, dtype)
	pcut.backward(tod, junk)
	wtod = tod*0+ivar[:,None]
	pcut.backward(wtod, junk)
	for sid in sids:
		shape, wcs = enmap.geometry(pos=[srcpos[::-1,sid]-R,srcpos[::-1,sid]+R], res=res, proj="car")
		area = enmap.zeros(shape, wcs, dtype)
		with bench.show("pmap"):
			pmap = pmat.PmatMap(scan, area)
		rhs  = enmap.zeros((3,)+shape, wcs, dtype)
		div  = rhs*0
		with bench.show("rhs"):
			pmap.backward(tod, rhs)
		with bench.show("div"):
			pmap.backward(wtod,div)
		div  = div[0]
		map  = rhs.copy()
		map[:,div>0] /= div[div>0]
		# Crop the outermost pixel, where outside hits will have accumulated
		map, div, area = [m[...,1:-1,1:-1] for m in [map,div,area]]
		# Find the local scanning velocity at the source position
		scan_vel = find_scan_vel(scan, srcpos[:,sid], aspeed)
		print scan_vel
		sdata.append(bunch.Bunch(
			map=map, div=div, srcpos=srcpos[:,sid], sid=sid,
			vel=scan_vel, fknee=args.fknee, alpha=args.alpha,
			id=id))

	write_sdata("%s/%s.hdf" % (args.odir, id), sdata)
	for i, sdat in enumerate(sdata):
		enmap.write_map("%s/%s_srcmap_%03d.fits" % (args.odir, id, sdat.sid), sdat.map)
