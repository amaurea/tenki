import numpy as np, os, time, h5py, astropy.io.fits
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
parser.add_argument("--acol", type=int, default=12)
parser.add_argument("-A", "--minamp",    type=float, default=0)
parser.add_argument("-f", "--fknee-mul", type=float, default=1.5)
parser.add_argument("-a", "--alpha",     type=float, default=5)
parser.add_argument("-R", "--radius",    type=float, default=10)
parser.add_argument("-r", "--res",       type=float, default=0.1)
parser.add_argument("-s", "--restrict",  type=str,   default=None)
args = parser.parse_args()

config.default("pmat_accuracy", 10.0, "Factor by which to lower accuracy requirement in pointing interpolation. 1.0 corresponds to 1e-3 pixels and 0.1 arc minute in polangle")

filedb.init()
ids  = filedb.scans[args.sel]
comm = mpi.COMM_WORLD
R    = args.radius*utils.arcmin
res  = args.res*utils.arcmin
dtype= np.float32
bsize_fknee = 100
bsize_ivar  = 400
utils.mkdir(args.odir)

rfreqs  = [12, 80]
drfreqs = [1,   1]

def find_scan_vel(scan, ipos, aspeed, dt=0.1):
	hpos  = coordinates.transform("equ","hor", ipos, time=scan.mjd0, site=scan.site)
	hpos[0] += aspeed*dt
	opos  = coordinates.transform("hor","equ", hpos, time=scan.mjd0, site=scan.site)
	opos[0] = utils.rewind(opos[0],ipos[0])
	return (opos-ipos)/dt

def bin_spectrum(ps, bsize):
	bisze = int(bsize)
	pix   = np.arange(ps.size)/bsize
	bps   = np.bincount(pix, ps)/np.bincount(pix)
	return bps

def measure_fknee(bps, df, fref=10, ratio=2):
	iref  = int(np.round(fref/df))
	above = np.where(bps[:iref] > ratio*bps[iref])[0]
	if len(above) == 0:
		print "This really shouldn't happen in measure_fknee"
		return df
	return above[-1]*df

def measure_power(ps, f, w, srate):
	i1 = int((f-w)*ps.shape[-1]/(srate/2.))
	i2 = int((f+w)*ps.shape[-1]/(srate/2.))
	return np.mean(ps[...,i1:i2]**-1,-1)**-1

def apply_bivar(tod, bivar, bsize, inplace=False):
	if not inplace: tod = tod.copy()
	nsamp = tod.shape[-1]
	work  = tod[...,:nsamp/bsize*bsize].reshape(tod.shape[:-1]+(-1,bsize))
	work *= bivar[...,:,None]
	# Handle left-over samples
	nleft = nsamp % bsize
	if nleft > 0: tod[...,-nleft:] *= bivar[...,-1,None]
	return tod

def read_srcs(fname, cols=(0,1,2)):
	if fname.endswith(".fits"):
		data = astropy.io.fits.open(fname)[1].data
		return np.array([data.ra*utils.degree,data.dec*utils.degree,data.sn])
	else:
		data = np.loadtxt(fname, usecols=cols).T
		data[:2] *= utils.degree
		return data

def write_sdata(ofile, sdata):
	# Output thumb for this tod
	with h5py.File(ofile, "w") as hfile:
		for i, sdat in enumerate(sdata):
			g = hfile.create_group("%d"%i)
			for key in ["map","div","srcpos","sid","vel","fknee","alpha",
					"id", "ctime", "dur", "el", "az", "off"]:
				g[key] = sdat[key]
			header = sdat.map.wcs.to_header()
			for key in header:
				g["wcs/"+key] = header[key]
			for key in sdat.site:
				g["site/"+key] = sdat.site[key]

bounds = filedb.scans.select(ids).data["bounds"]

# Load source database
srcdata  = read_srcs(args.srclist, cols=(args.rcol, args.dcol, args.acol))
srcpos, amps = srcdata[:2], srcdata[2]
print srcpos, amps
allowed  = set(range(amps.size))
allowed &= set(np.where(amps > args.minamp)[0])
if args.restrict is not None:
	selected = [int(w) for w in args.restrict.split(",")]
	allowed &= set(selected)

for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	# Check if we hit any of the sources. We first make sure
	# there's no angle wraps in the bounds, and then move the sources
	# to the same side of the sky.
	poly      = bounds[:,:,ind]*utils.degree
	poly[0]   = utils.rewind(poly[0],poly[0,0])
	srcpos    = srcpos.copy()
	srcpos[0] = utils.rewind(srcpos[0], poly[0,0])
	sids      = np.where(utils.point_in_polygon(srcpos.T, poly.T))[0]
	sids      = sorted(list(set(sids)&allowed))
	if len(sids) == 0:
		print "%s has 0 srcs: skipping" % id
		continue
	print "%s has %d srcs: %s" % (id,len(sids),",".join([str(i) for i in sids]))
	entry = filedb.data[id]
	try:
		with bench.mark("read"):
			d = actdata.read(entry)
		with bench.mark("calib"):
			d = actdata.calibrate(d, exclude=["autocut"])
	except errors.DataMissing as e:
		print "%s skipped: %s" % (id, e.message)
		continue
	tod = d.tod.astype(dtype)
	del d.tod
	# Apply high-pass filter. Will assume white tod after this
	with bench.mark("filter"):
		freqs = fft.rfftfreq(d.nsamp)*d.srate
		ft    = fft.rfft(tod)
		ps    = np.abs(ft)**2
		rpows = [measure_power(ps,rfreq,drfreq,d.srate) for rfreq,drfreq in zip(rfreqs, drfreqs)]
		rpows = np.array(rpows)
		# Determine the fknee to use. First get a typical spectrum.
		# This does not work well with s16, which currently doesn't
		# have time constants.
		ps     = np.median(ps,0)
		bps    = bin_spectrum(ps, bsize_fknee)
		fknee  = measure_fknee(bps, d.srate/2/ps.size*bsize_fknee)
		print "fknee %7.4f" % fknee
		#np.savetxt("ps.txt", ps)
		#1/0
		fknee *= args.fknee_mul
		ft[:,0]   = 0
		ft[:,1:] /= 1 + (freqs[1:]/fknee)**-args.alpha
		fft.ifft(ft, tod, normalize=True)
		del ft

	# Estimate white noise level in bins, and weight tod by it
	ivar  = 1/np.mean(tod**2,-1)
	bivar = 1/np.mean(tod[:,:d.nsamp/bsize_ivar*bsize_ivar].reshape(d.ndet,-1,bsize_ivar)**2,-1)
	tod   = apply_bivar(tod, bivar, bsize_ivar, inplace=True)
	#tod  *= ivar[:,None]

	# Find azimuth scanning speed
	aspeed = np.median(np.abs(d.boresight[1,1:]-d.boresight[1,:-1])[::10])*d.srate
	tref   = d.boresight[0,d.nsamp/2]

	# Build a small, high-res map around each source
	sdata = []
	with bench.mark("scan"):
		scan = actscan.ACTScan(entry, d=d)
	pcut = pmat.PmatCut(scan)
	junk = np.zeros(pcut.njunk, dtype)
	pcut.backward(tod, junk)
	wtod = apply_bivar(tod*0+1,bivar,bsize_ivar,inplace=True)
	#wtod = tod*0+ivar[:,None]
	pcut.backward(wtod, junk)
	for sid in sids:
		shape, wcs = enmap.geometry(pos=[srcpos[::-1,sid]-R,srcpos[::-1,sid]+R], res=res, proj="car")
		area = enmap.zeros(shape, wcs, dtype)
		with bench.mark("pmap"):
			pmap = pmat.PmatMap(scan, area)
		rhs  = enmap.zeros((3,)+shape, wcs, dtype)
		div  = rhs*0
		with bench.mark("rhs"):
			pmap.backward(tod, rhs)
		with bench.mark("div"):
			pmap.backward(wtod,div)
		div  = div[0]
		map  = rhs.copy()
		map[:,div>0] /= div[div>0]
		map = map[0]
		# Crop the outermost pixel, where outside hits will have accumulated
		map, div, area = [m[...,1:-1,1:-1] for m in [map,div,area]]
		# Find the local scanning velocity at the source position
		scan_vel = find_scan_vel(scan, srcpos[:,sid], aspeed)
		hbox = scan.box
		dur  = d.nsamp*d.srate
		el   = np.mean(scan.box[:,2])
		az   = scan.box[:,1]
		sdata.append(bunch.Bunch(
			map=map, div=div, srcpos=srcpos[:,sid], sid=sid,
			vel=scan_vel, fknee=fknee, alpha=args.alpha,
			id=id, ctime=tref, dur=dur, el=el, az=az,
			site=d.site, off=d.point_correction))
	del tod, wtod, d

	write_sdata("%s/%s.hdf" % (args.odir, id), sdata)
	for i, sdat in enumerate(sdata):
		enmap.write_map("%s/%s_srcmap_%03d.fits" % (args.odir, id, sdat.sid), sdat.map)
