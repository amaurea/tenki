from __future__ import division, print_function
import numpy as np, argparse, enlib.scan, os
from enlib import enmap, utils, config, scansim, log, powspec, fft, bunch
from enact import actscan, filedb, nmat_measure

config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")

parser = config.ArgumentParser()
parser.add_argument("odir")
parser.add_argument("--area",  type=str)
parser.add_argument("--bore",  type=str, default="grid:2:0.2:0.8")
parser.add_argument("--dets",  type=str, default="scattered:3:3:2.0")
parser.add_argument("--signal",type=str, default="ptsrc:100:1e3:-3")
parser.add_argument("--noise", type=str, default="1/f:20:2:0.5")
parser.add_argument("--seed",  type=int, default=1)
parser.add_argument("--measure", type=float, default=None)
parser.add_argument("--real",  type=str, default=None)
args = parser.parse_args()

log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level)
utils.mkdir(args.odir)

if args.area:
	area = enmap.read_map(args.area)
	if area.ndim == 2: area = area[None]
else:
	shape, wcs = enmap.geometry(pos=np.array([[-1,-1],[1,1]])*np.pi/180, shape=(600,600), pre=(3,), proj="car", ref=[0,0])
	area = enmap.zeros(shape, wcs)

def get_scans(area, signal, bore, dets, noise, seed=0, real=None, noise_override=None):
	scans = []
	# Get real scan information if necessary
	L.debug("real")
	if real:
		real_scans = []
		filedb.init()
		db   = filedb.data
		ids  = fileb.scans[real].ids
		for id in ids:
			try:
				real_scans.append(actscan.ACTScan(db[id]))
			except errors.DataMissing as e:
				L.debug("Skipped %s (%s)" % (id, str(e)))
	##### Dets #####
	L.debug("dets")
	sim_dets = []
	toks = dets.split(":")
	if toks[0] == "scattered":
		ngroup, nper, rad = int(toks[1]), int(toks[2]), float(toks[3])
		sim_dets = [scansim.dets_scattered(ngroup, nper,rad=rad*np.pi/180/60)]
		margin   = rad*np.pi/180/60
	elif toks[0] == "row":
		ngroup, nper, rad = int(toks[1]), int(toks[2]), float(toks[3])
		sim_dets = [scansim.dets_row(ngroup, nper, rad=rad*np.pi/180/60)]
		margin   = rad*np.pi/180/60
	elif toks[0] == "real":
		ndet = int(toks[1])
		nper = 1 # FIXME
		dslice = slice(0,ndet) if ndet > 0 else slice(None)
		sim_dets = [bunch.Bunch(comps=s.comps[dslice], offsets=s.offsets[dslice]) for s in real_scans]
		margin = np.max([np.sum(s.offsets**2,1)**0.5 for s in sim_dets])
	else: raise ValueError
	#### Boresight. Determines our number of scans ####
	L.debug("bore")
	sim_bore = []
	toks = bore.split(":")
	if toks[0] == "grid":
		nscan, density, short = int(toks[1]), float(toks[2]), float(toks[3])
		for i in range(nscan):
			tbox = shorten(area.box(),i%2,short)
			sim_bore.append(scansim.scan_grid(tbox, density*np.pi/180/60, dir=i, margin=margin))
	elif toks[0] == "ces":
		nscan = int(toks[1])
		azs   = [float(w)*utils.degree for w in toks[2].split(",")]
		els   = [float(w)*utils.degree for w in toks[3].split(",")]
		mjd0  = float(toks[4])
		dur   = float(toks[5])
		azrate= float(toks[6]) if len(toks) > 6 else 1.5*utils.degree
		srate = float(toks[7]) if len(toks) > 7 else 400
		nsamp = utils.nint(dur*srate)
		for i in range(nscan):
			mjd  = mjd0 + dur*(i//(2*len(els)))/(24*3600)
			el   = els[(i//2)%len(els)]
			az1, az2 = azs
			if i%2 == 1: az1, az2 = -az2, -az1
			box = np.array([[az1,el],[az2,el]])
			sim_bore.append(scansim.scan_ceslike(nsamp, box, mjd0=mjd, srate=srate, azrate=azrate))
	elif toks[0] == "real":
		sim_bore = [bunch.Bunch(boresight=s.boresight, hwp_phase=s.hwp_phase, sys=s.sys, site=s.site, mjd0=s.mjd0) for s in real_scans]
	else: raise ValueError
	nsim = len(sim_bore)
	# Make one det info per scan
	sim_dets = sim_dets*(nsim//len(sim_dets))+sim_dets[:nsim%len(sim_dets)]
	#### Noise ####
	L.debug("noise")
	sim_nmat = []
	toks = noise.split(":")
	nonoise = False
	if toks[0] == "1/f":
		sigma, alpha, fknee = [float(v) for v in toks[1:4]]
		nonoise = sigma < 0
		for i in range(nsim):
			sim_nmat.append(scansim.oneoverf_noise(sim_dets[i].comps.shape[0], sim_bore[i].boresight.shape[0], sigma=np.abs(sigma), alpha=alpha, fknee=fknee))
	elif toks[0] == "detcorr":
		sigma, alpha, fknee = [float(v) for v in toks[1:4]]
		nmode = int(toks[4])
		nonoise = sigma < 0
		for i in range(nsim):
			sim_nmat.append(scansim.oneoverf_detcorr_noise(ndet=sim_dets[i].comps.shape[0], nper=nper, nsamp=sim_bore[i].boresight.shape[0], sigma=np.abs(sigma), alpha=alpha, fknee=fknee, nmode=nmode))
	elif toks[0] == "real":
		scale = 1.0 if len(toks) < 2 else float(toks[1])
		for i,s in enumerate(real_scans):
			ndet = len(sim_dets[i].offsets)
			nmat = s.noise[:ndet]*scale**-2
			sim_nmat.append(nmat)
	else: raise ValueError
	noise_scale = not nonoise if noise_override is None else noise_override
	sim_nmat = sim_nmat*(nsim//len(sim_nmat))+sim_nmat[:nsim%len(sim_nmat)]
	### Signal ####
	L.debug("signal")
	toks = signal.split(":")
	if toks[0] == "none":
		for i in range(nsim):
			scans.append(scansim.SimPlain(sim_bore[i], sim_dets[i], sim_nmat[i], seed=seed+i, noise_scale=noise_scale))
	elif toks[0] == "ptsrc":
		# This one always operates in the same coordinates as 
		nsrc, amp, fwhm = int(toks[1]), float(toks[2]), float(toks[3])
		np.random.seed(seed)
		sim_srcs = scansim.rand_srcs(area.box(), nsrc, amp, abs(fwhm)*np.pi/180/60, rand_fwhm=fwhm<0)
		for i in range(nsim):
			scans.append(scansim.SimSrcs(sim_bore[i], sim_dets[i], sim_srcs, sim_nmat[i], seed=seed+i, noise_scale=noise_scale))
	elif toks[0] == "vsrc":
		# Create a single variable source
		ra, dec, fwhm = float(toks[1])*np.pi/180, float(toks[2])*np.pi/180, float(toks[3])*np.pi/180/60
		amps = [float(t) for t in toks[4].split(",")]
		for i in range(nsim):
			sim_srcs = bunch.Bunch(pos=np.array([[dec,ra]]),amps=np.array([[amps[i],0,0,0]]), beam=np.array([fwhm/(8*np.log(2)**0.5)]))
			scans.append(scansim.SimSrcs(sim_bore[i], sim_dets[i], sim_srcs, sim_nmat[i], seed=seed+i, noise_scale=noise_scale, nsigma=20))
	elif toks[0] == "cmb":
		np.random.seed(seed)
		ps = powspec.read_spectrum(toks[1])
		sim_map  = enmap.rand_map(area.shape, area.wcs, ps)
		for i in range(nsim):
			scans.append(scansim.SimMap(sim_bore[i], sim_dets[i], sim_map,    sim_nmat[i], seed=seed+i, noise_scale=noise_scale))
	else: raise ValueError
	return scans

def shorten(box, i, s=0.5):
	box = np.array(box)
	m   = (box[1]+box[0])/2
	w   = (box[1]-box[0])/2*s
	j   = i%2
	box[:,j] = [m[j]-w[j],m[j]+w[j]]
	return box

def get_model(s, area):
	pos = area.posmap().reshape(2,-1)[::-1].T
	model = np.rollaxis(s.get_model(pos),-1).reshape(-1,area.shape[1],area.shape[2])
	return enmap.ndmap(model, area.wcs)[:area.shape[0]]

if args.measure is None:
	scans = get_scans(area, args.signal, args.bore, args.dets, args.noise, seed=args.seed, real=args.real)
else:
	# Build noise model the same way we do for the real data, i.e. based on
	# measuring data itself. But do that based on a version with more noise
	# than the real one, to simulate realistic S/N ratios without needing
	# too many samples
	scans = get_scans(area, args.signal, args.bore, args.dets, args.noise, seed=args.seed, real=args.real, noise_override=args.measure)
	nmats = []
	for scan in scans:
		ft = fft.rfft(scan.get_samples()) * scan.nsamp**-0.5
		nmats.append(nmat_measure.detvecs_jon(ft, 400.0, shared=True))
	scans = get_scans(area, args.signal, args.bore, args.dets, args.noise, seed=args.seed, real=args.real)
	for scan,nmat in zip(scans,nmats):
		scan.noise = nmat

enmap.write_map(args.odir + "/area.fits", area)
model = get_model(scans[0], area)
enmap.write_map(args.odir + "/model.fits", model)
with open(args.odir + "/tods.txt", "w") as ofile:
	for i, scan in enumerate(scans):
		L.info("scan %2d/%d" % (i+1,len(scans)))
		enlib.scan.write_scan(args.odir + "/scan%03d.hdf" % i, scan)
		ofile.write("%s/scan%03d.hdf\n" % (args.odir, i))
