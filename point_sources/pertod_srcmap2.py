import numpy as np, sys, os, ephem
from enlib import utils
with utils.nowarn(): import h5py
from enlib import config, pmat, mpi, errors, gapfill, enmap, bench, ephemeris
from enlib import fft, array_ops, sampcut, cg
from enact import filedb, actscan, actdata, cuts, nmat_measure
from astropy.io import fits
config.set("pmat_cut_type",  "full")
parser = config.ArgumentParser()
parser.add_argument("sel")
parser.add_argument("srcs")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?")
parser.add_argument("-R", "--dist",   type=float, default=4)
parser.add_argument("-y", "--ypad",   type=float, default=3)
parser.add_argument("-s", "--src",    type=int,   default=None, help="Only analyze given source")
parser.add_argument("-m", "--model",  type=str,   default="constrained")
parser.add_argument("-o", "--output", type=str,   default="individual")
parser.add_argument("--hit-tol",      type=float, default=0.5)
parser.add_argument("-c", "--cont",   action="store_true")
args = parser.parse_args()

comm = mpi.COMM_WORLD
filedb.init()
R    = args.dist * utils.arcmin
ypad = args.ypad * utils.arcmin
csize= 100
config.set("pmat_accuracy", 25)
config.set("pmat_ptsrc_cell_res", 2*(R+ypad)/utils.arcmin)
config.set("pmat_ptsrc_rsigma", 5)
config.set("pmat_interpol_pad", 5+ypad/utils.arcmin)
osys = config.get("tod_sys")

dtype = np.float32
area  = enmap.read_map(args.area).astype(dtype)
ncomp = 3
shape = area.shape[-2:]
utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.tag:  prefix += args.tag + "_"

# Set up a dummy beam that represents our source mask. It will
# go linearly from 1 to 0 at 2R, letting us use 0.5 as the 1R cutoff.
beam = np.array([[0,1],[2*R,0]]).T

src_dtype = [("ra","f"),("dec","f"),("type","S20"),("name","S20")]
def load_srcs(desc):
	# Reads in the set of sources to map, returning
	# [nsrc,{ra,dec,type,name}]. type will be "fixed"
	# for fixed sources and "planet" for moving sources.
	# Moving sources will have their coordinates updated for
	# each TOD.
	if desc in ephem.__dict__:
		res = np.zeros(1, src_dtype).view(np.recarray)
		res.type = "planet"
		res.name = desc
		return res
	try:
		obj  = ephemeris.read_object(desc)
		name = "obj000"
		setattr(ephem, name, lambda: obj)
		res = np.zeros(1, src_dtype).view(np.recarray)
		res.type = "planet"
		res.name = name
		return res
	except (IOError, OSError):
		srcs = np.loadtxt(args.srcs, usecols=[0,1], ndmin=2)
		res  = np.zeros(len(srcs), src_dtype).view(np.recarray)
		res.ra, res.dec = srcs.T[:2]*utils.degree
		res.type = "fixed"
		return res

srcs = load_srcs(args.srcs)

# Find out which sources are hit by each tod
db = filedb.scans.select(filedb.scans[args.sel])
tod_srcs = {}
for sid, src in enumerate(srcs):
	if args.src is not None and sid != args.src: continue
	if src.type == "planet":
		# This is a bit hacky, but sometimes the "t" member is unavailable
		# in the database. This also ignores the planet movement during the
		# TOD. We will take that into account in the final step in the mapping, though.
		t = np.char.partition(db.ids,".")[:,0].astype(float)+300
		ra, dec = ephemeris.ephem_pos(src.name, utils.ctime2mjd(t), dt=0)[:2]
	else: ra, dec = src.ra, src.dec
	points = np.array([ra, dec])
	polys  = db.data["bounds"]*utils.degree
	polys[0] = utils.rewind(polys[0], points[0])
	polys[0] = utils.rewind(polys[0], polys[0,0])
	inside   = utils.point_in_polygon(points.T, polys.T)
	dists    = utils.poly_edge_dist(points.T, polys.T)
	dists    = np.where(inside, 0, dists)
	hit      = np.where(dists < args.hit_tol*utils.degree)[0]
	for id in db.ids[hit]:
		if not id in tod_srcs: tod_srcs[id] = []
		tod_srcs[id].append(sid)

# Prune those those that are done
if args.cont:
	good = []
	for id in tod_srcs:
		bid = id.replace(":","_")
		ndone = 0
		for sid in tod_srcs[id]:
			if os.path.exists("%s%s_src%03d_map.fits" % (prefix, bid, sid)) or os.path.exists("%s%s_empty.txt" % (prefix, bid)):
				ndone += 1
		if ndone < len(tod_srcs[id]):
			good.append(id)
	tod_srcs = {id:tod_srcs[id] for id in good}
ids = sorted(tod_srcs.keys())

def smooth(tod, srate, fknee=10, alpha=10):
	ft   = fft.rfft(tod)
	freq = fft.rfftfreq(tod.shape[-1])*srate
	flt  = 1/(1+(freq/fknee)**alpha)
	ft  *= flt
	fft.ifft(ft, tod, normalize=True)
	return tod

def calc_model_joneig(tod, cut, srate=400):
	return smooth(gapfill.gapfill_joneig(tod, cut, inplace=False), srate)

def calc_model_constrained(tod, cut, srate=400, mask_scale=0.3, lim=3e-4, maxiter=50, verbose=False):
	# First do some simple gapfilling to avoid messing up the noise model
	tod = sampcut.gapfill_linear(cut, tod, inplace=False)
	ft = fft.rfft(tod) * tod.shape[1]**-0.5
	iN = nmat_measure.detvecs_jon(ft, srate)
	del ft
	iV = iN.ivar*mask_scale
	def A(x):
		x   = x.reshape(tod.shape)
		Ax  = iN.apply(x.copy())
		Ax += sampcut.gapfill_const(cut, x*iV[:,None], 0, inplace=True)
		return Ax.reshape(-1)
	b  = sampcut.gapfill_const(cut, tod*iV[:,None], 0, inplace=True).reshape(-1)
	x0 = sampcut.gapfill_linear(cut, tod).reshape(-1)
	solver = cg.CG(A, b, x0)
	while solver.i < maxiter and solver.err > lim:
		solver.step()
		if verbose:
			print "%5d %15.7e" % (solver.i, solver.err)
	return solver.x.reshape(tod.shape)

def map_to_header(map):
	header = map.wcs.to_header(relax=True)
	# Add our map headers
	header['NAXIS'] = map.ndim
	for i,n in enumerate(map.shape[::-1]):
		header['NAXIS%d'%(i+1)] = n
	return header

def write_package(fname, maps, divs, src_ids, d):
	header = map_to_header(maps)
	header["id"] = d.entry.id + ":" + entry.tag
	header["off_x"] = d.point_correction[0]/utils.arcmin
	header["off_y"] = d.point_correction[1]/utils.arcmin
	header["t1"]     = d.boresight[0,0]
	header["t2"]     = d.boresight[0,-1]
	meanoff = np.mean(d.point_offset,0)/utils.degree
	header["az1"]    = np.min(d.boresight[1,:])/utils.degree + meanoff[0]
	header["az2"]    = np.max(d.boresight[1,:])/utils.degree + meanoff[1]
	header["el"]     = np.mean(d.boresight[2,::100])/utils.degree

	hdu_maps = astropy.io.fits.PrimaryHDU(maps, header)
	hdu_divs = astropy.io.fits.ImageHDU(divs, map_to_header(divs), name="div"),
	hdu_ids  = astropy.io.fits.TableHDU(src_ids, name="ids")

	hdus = astropy.io.fits.HDUList([hdu_maps, hdu_divs, hdu_ids])
	with utils.nowarn():
		hdus.writeto(fname, clobber=True)

for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	print "A", id, comm.rank
	bid   = id.replace(":","_")
	entry = filedb.data[id]
	# Read the tod as usual
	try:
		with bench.mark("read"):
			d = actdata.read(entry)
		with bench.mark("calibrate"):
			d = actdata.calibrate(d, exclude=["autocut"])
		# Replace the beam with our dummy beam
		d.beam = beam
		if d.ndet < 2 or d.nsamp < 2: raise errors.DataMissing("no data in tod")
	except errors.DataMissing as e:
		print "Skipping %s (%s)" % (id, comm.rank, e.args[0])
		# Make a dummy output file so we can skip this tod in the future
		with open("%s%s_empty.txt" % (prefix, bid),"w"): pass
		continue
	print "%3d Processing %s [ndet:%d, nsamp:%d, nsrc:%d]" % (comm.rank, id, d.ndet, d.nsamp, len(tod_srcs[id]))
	print "B", id, comm.rank
	# Fill in representative ra, dec for planets for this tod
	for sid in np.where(srcs.type == "planet")[0]:
		srcs.ra[sid], srcs.dec[sid] = ephemeris.ephem_pos(srcs.name[sid], utils.ctime2mjd(d.boresight[0,d.nsamp//2]), dt=0)[:2]
	# Very simple white noise model. This breaks if the beam has been tod-smoothed by this point.
	print "C", id, comm.rank
	with bench.mark("ivar"):
		tod  = d.tod
		del d.tod
		tod -= np.mean(tod,1)[:,None]
		tod  = tod.astype(dtype)
		diff = tod[:,2:]-tod[:,:-2]
		diff = diff[:,:diff.shape[-1]/csize*csize].reshape(d.ndet,-1,csize)
		ivar = 1/(np.median(np.mean(diff**2,-1),-1)/2**0.5)
		del diff
	print "D", id, comm.rank
	with bench.mark("actscan"):
		scan = actscan.ACTScan(entry, d=d)
	print "E", id, comm.rank
	with bench.mark("pmat1"):
		# Build effective source parameters for this TOD. This will ignore planet motion,
		# but this should only be a problem for very near objects like the moon or ISS
		src_param = np.zeros((len(srcs),8))
		src_param[:,0]   = srcs.dec
		src_param[:,1]   = srcs.ra
		src_param[:,2]   = 1
		src_param[:,5:7] = 1
		# And use it to build our source model projector. This is only used for the cuts
		psrc = pmat.PmatPtsrc(scan, src_param)
	print "F", id, comm.rank
	with bench.mark("source mask"):
		# Find the samples where the sources live
		src_mask = np.zeros(tod.shape, np.bool)
		# Allow elongating the mask vertically
		nypad = 1
		dypad = R/2
		if ypad > 0:
			nypad = int((2*ypad)//dypad)+1
			dypad = (2*ypad)/(nypad-1)
		# Hack: modify detector offsets to apply the y broadening
		detoff = scan.offsets.copy()
		src_tod = tod*0
		for yi in range(nypad):
			yoff    = -ypad + yi*dypad
			psrc.scan.offsets = detoff.copy()
			psrc.scan.offsets[:,2] -= yoff
			psrc.forward(src_tod, src_param, tmul=0)
			src_mask |= src_tod > 0.5
		del src_tod
		# Undo the hack here
		psrc.scan.offsets = detoff
		src_cut  = sampcut.from_mask(src_mask)
		src_cut *= scan.cut
		del src_mask
	print "G", id, comm.rank
	try:
		with bench.mark("atm model"):
			if   args.model == "joneig":
				model = calc_model_joneig(tod, src_cut, d.srate)
			elif args.model == "constrained":
				model = calc_model_constrained(tod, src_cut, d.srate, verbose=True)
	except np.linalg.LinAlgError as e:
		print "%3d %s Error building noide model: %s" % (comm.rank, id, e.args[0])
		continue
	with bench.mark("atm subtract"):
		tod -= model
		del model
		tod  = tod.astype(dtype, copy=False)
	# Should now be reasonably clean of correlated noise.
	# Proceed to make simple binned map for each point source. We need a separate
	# pointing matrix for each because each has its own local coordinate system.
	print "H", id, comm.rank
	tod *= ivar[:,None]
	sampcut.gapfill_const(scan.cut, tod, inplace=True)
	nsrc  = len(tod_srcs[id])
	omaps = enmap.zeros((nsrc,ncomp)+shape, area.wcs, dtype)
	odivs = enmap.zeros((nsrc,ncomp,ncomp)+shape, area.wcs, dtype)
	print "I", id, comm.rank
	for si, sid in enumerate(tod_srcs[id]):
		src  = srcs[sid]
		if   src.type == "fixed":  sys = "%s:%.6f_%.6f:cel/0_0:%s" % (osys, src.ra/utils.degree, src.dec/utils.degree, osys)
		elif src.type == "planet": sys = "%s:%s/0_0" % (osys, src.name)
		else: raise ValueError("Invalid source type '%s'" % src.type)
		rhs  = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
		div  = enmap.zeros((ncomp,ncomp)+shape, area.wcs, dtype)
		with bench.mark("pmat %s" % sid):
			pmap = pmat.PmatMap(scan, area, sys=sys)
		with bench.mark("rhs %s" % sid):
			pmap.backward(tod, rhs)
		with bench.mark("hits"):
			for i in range(ncomp):
				div[i,i] = 1
				pmap.forward(tod, div[i])
				tod *= ivar[:,None]
				sampcut.gapfill_const(scan.cut, tod, inplace=True)
				div[i] = 0
				pmap.backward(tod, div[i])
		with bench.mark("map %s" % sid):
			idiv = array_ops.eigpow(div, -1, axes=[0,1], lim=1e-5, fallback="scalar")
			map  = enmap.map_mul(idiv, rhs)
		omaps[si] = map
		odivs[si] = div
		del rhs, div, idiv, map
	print "J", id, comm.rank
	# Write out the resulting maps
	if args.output == "individual":
		with bench.mark("write"):
			for si, sid in enumerate(tod_srcs[id]):
				print "K", id, comm.rank, sid
				enmap.write_map("%s%s_src%03d_map.fits" % (prefix, bid, sid), omaps[si])
				enmap.write_map("%s%s_src%03d_div.fits" % (prefix, bid, sid), odivs[si])
				#enmap.write_map("%s%s_src%03d_rhs.fits" % (prefix, bid, sid), rhs)
	elif args.output == "grouped":
		with bench.mark("write"):
			write_package("%s%s.fits" % (prefix, bid), omaps, odivs, tod_srcs[id], d)
	del d, scan, pmap, tod, omaps, odivs

comm.Barrier()
