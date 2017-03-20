import numpy as np, os, time, h5py, astropy.io.fits, sys, argparse
from scipy import optimize
from enlib import utils, mpi, fft, enmap, bunch, coordinates
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
parser.add_argument("-b", "--fwhm",  type=float, default=1.3)
parser.add_argument("-x", "--ignore-ctime", action="store_true")
parser.add_argument("--orad",        type=float, default=10)
parser.add_argument("--ores",        type=float, default=0.1)
args = parser.parse_args()

comm = mpi.COMM_WORLD
utils.mkdir(args.odir)
fwhm = args.fwhm * utils.arcmin

def fixorder(res):
	if res.dtype.byteorder not in ['=','<' if sys.byteorder == 'little' else '>']:
		res = res.byteswap().newbyteorder()
	return res

def read_sdata(ifile):
	# Output thumb for this tod
	with h5py.File(ifile, "r") as hfile:
		sdata = [None for key in hfile]
		for key in hfile:
			ind  = int(key)
			g    = hfile[key]
			hwcs = g["wcs"]
			header = astropy.io.fits.Header()
			for key in hwcs:
				header[key] = hwcs[key].value
			wcs = enmap.enlib.wcs.WCS(header).sub(2)
			sdata[ind] = bunch.Bunch(
				map = enmap.ndmap(fixorder(g["map"].value), wcs),
				div = enmap.ndmap(fixorder(g["div"].value), wcs),
				sid = g["sid"].value,
				id  = g["id"].value,
				vel = g["vel"].value,
				fknee = g["fknee"].value,
				alpha = g["alpha"].value,
				srcpos = g["srcpos"].value,
				ctime = g["ctime"].value if "ctime" in g and not args.ignore_ctime else float(g["id"].value.split(".")[0]),
				)
	return sdata

class BeamModel:
	def __init__(self, fwhm, vel, dec_ref, fknee, alpha, nsigma=2000, nsub=100, order=3):
		# Vel is [dra,ddec]/s.
		sigma  = fwhm/(8*np.log(2))**0.5
		res    = sigma/nsub
		vel    = np.array(vel)
		vel[0]*= np.cos(dec_ref)
		speed  = np.sum(vel**2)**0.5
		# Build coordinate system along velocity
		npoint = 2*nsigma*nsub
		x      = (np.arange(npoint)-npoint/2)*res
		# Build the beam along the velocity
		vbeam  = np.exp(-0.5*(x**2/sigma**2))
		# Apply fourier filter. Our angular step size is res radians. This
		# corresponds to a time step of res/speed in seconds.
		fbeam  = fft.rfft(vbeam)
		freq   = fft.rfftfreq(npoint, res/speed)
		fbeam[1:] /= 1 + (freq[1:]/fknee)**-alpha
		vbeam  = fft.ifft(fbeam, vbeam, normalize=True)
		# Prefilter for fast lookups
		vbeam  = utils.interpol_prefilter(vbeam, npre=0, order=order)
		# The total beam will be this beam times a normal one in the
		# perpendicular direction.
		self.dec_ref = dec_ref
		self.e_para  = vel/np.sum(vel**2)**0.5
		self.e_orto  = np.array([-self.e_para[1],self.e_para[0]])
		self.sigma   = sigma
		self.res     = res
		self.vbeam   = vbeam
		self.order   = order
		# Not really necessary to store these
		self.fwhm  = fwhm
		self.vel   = vel # physical
		self.fknee = fknee
		self.alpha = alpha
	def eval(self, rvec):
		"""Evaluate beam at positions rvec[{dra,dec},...] relative to beam center"""
		# Decompose into parallel and orthogonal parts
		rvec     = np.asanyarray(rvec).copy()
		rvec[0] *= np.cos(self.dec_ref)
		rpara    = np.sum(rvec*self.e_para[:,None,None],0)
		rorto    = np.sum(rvec*self.e_orto[:,None,None],0)
		# Evaluate each beam component
		ipara    = rpara/self.res+self.vbeam.size/2
		bpara    = utils.interpol(self.vbeam, ipara[None], mask_nan=False, order=self.order, prefilter=False)
		borto    = np.exp(-0.5*rorto**2/self.sigma**2)
		res      = enmap.samewcs(bpara*borto, rvec)
		return res

# Single-source likelihood evaluator
class Srclik:
	def __init__(self, map, div, beam, maxdist=8):
		self.map   = map.preflat[0]
		#enmap.write_map("map.fits", map)
		self.div   = div
		self.posmap= map.posmap()
		self.off   = map.size/3
		#enmap.write_map("posmap.fits", self.posmap)
		self.beam  = beam
	def calc_profile(self, pos):
		dpos = self.posmap[::-1] - pos[:,None,None]
		return self.beam.eval(dpos)
	def calc_amp(self, profile):
		ivamp= np.sum(profile**2*self.div)
		if ivamp == 0: return 0, np.inf
		vamp = 1/ivamp
		amp  = vamp*np.sum(profile*self.div*self.map)
		if ~np.isfinite(amp): amp = 0
		return amp, vamp
	def calc_model(self, pos):
		profile  = self.calc_profile(pos)
		amp, vamp= self.calc_amp(profile)
		return profile*amp
		return profile*np.abs(amp)
	def calc_chisq(self, posoff):
		model = self.calc_model(posoff)
		resid = self.map - model
		chisq = np.sum(resid**2*self.div)
		return chisq
	def simple_max(self):
		smap = enmap.smooth_gauss(self.map*self.div**0.5, self.beam.sigma)
		pos  = enmap.argmax(smap)
		val  = smap.at(pos)
		return pos[::-1], val

class DposTrans:
	def __init__(self, ref_pos_cel, ctime):
		self.ref_pos_cel = ref_pos_cel
		self.mjd         = utils.ctime2mjd(ctime)
	def to_cel(self, dpos): return dpos
	def from_cel(self, dpos): return dpos

class DposTransFoc(DposTrans):
	def __init__(self, ref_pos_cel, ctime):
		self.ref_pos_cel = ref_pos_cel
		self.mjd          = utils.ctime2mjd(ctime)
		self.ref_pos_hor = coordinates.transform("cel","hor",ref_pos_cel,time=self.mjd)
	def to_cel(self, dpos):
			pos_hor = coordinates.recenter(dpos, [0,0,self.ref_pos_hor[0],self.ref_pos_hor[1]])
			dpos_cel = utils.rewind(
					coordinates.transform("hor","cel",pos_hor,time=self.mjd)-self.ref_pos_cel)
			return dpos_cel
	def from_cel(self, dpos):
		pos_hor = coordinates.transform("cel","hor",self.ref_pos_cel+dpos,time=self.mjd)
		pos_foc = utils.rewind(coordinates.recenter(pos_hor,[self.ref_pos_hor[0],self.ref_pos_hor[1],0,0]))
		return pos_foc

class SrcFitter:
	def __init__(self, sdata, fwhm, ctrans=DposTransFoc):
		self.nsrc  = len(sdata)
		self.scale = 6*utils.arcmin
		self.rmax  = 6*utils.arcmin
		self.ngrid = 15
		self.sdata = sdata
		self.liks  = []
		for s in sdata:
			beam = BeamModel(fwhm, s.vel, s.srcpos[1], s.fknee, s.alpha)
			lik  = Srclik(s.map, s.div, beam)
			self.liks.append(lik)
		if ctrans is None: ctrans = lambda (dpos,sdat): dpos
		self.trfs  = [ctrans(s.srcpos, s.ctime) for s in sdata]
		self.i     = 0
		self.verbose = False
	def calc_chisq_wrapper(self, x):
		dpos  = x*self.scale
		chisq = self.calc_chisq(dpos)
		if self.verbose:
			print "%4d %9.4f %9.4f %15.7e" % (self.i, dpos[0]/utils.arcmin, dpos[1]/utils.arcmin, chisq)
		self.i += 1
		return chisq
	def calc_chisq(self, dpos):
		chisqs = []
		for lik, trf, s in zip(self.liks, self.trfs, self.sdata):
			dpos_cel = trf.to_cel(dpos)
			chisqs.append(lik.calc_chisq(s.srcpos+dpos_cel))
		chisq  = np.sum(chisqs)
		rrel   = np.sum(dpos**2)**0.5/self.rmax
		if rrel > 1: chisq *= rrel
		return chisq
	def calc_deriv(self, dpos, step=0.05*utils.arcmin):
		return np.array([
			self.calc_chisq(dpos+[step,0])-self.calc_chisq(dpos-[step,0]),
			self.calc_chisq(dpos+[0,step])-self.calc_chisq(dpos-[0,step])])/(2*step)
	def calc_hessian(self, dpos, step=0.05*utils.arcmin):
		return np.array([
			self.calc_deriv(dpos+[step,0])-self.calc_deriv(dpos-[step,0]),
			self.calc_deriv(dpos+[0,step])-self.calc_deriv(dpos-[0,step])
		])/(2*step)
	def calc_full_result(self, dpos):
		amps, models, poss_cel, poss_hor, vamps = [], [], [], [], []
		for i in range(self.nsrc):
			lik, sd = self.liks[i], self.sdata[i]
			dpos_cel= self.trfs[i].to_cel(dpos)
			pos_cel = dpos_cel + sd.srcpos
			pos_hor = coordinates.transform("cel","hor",pos_cel,utils.ctime2mjd(self.sdata[i].ctime))
			profile = lik.calc_profile(pos_cel)
			amp, vamp = lik.calc_amp(profile)
			amps.append(amp)
			vamps.append(vamp)
			models.append(amp*profile)
			poss_cel.append(pos_cel)
			poss_hor.append(pos_hor)
		# Get the position uncertainty
		hess  = self.calc_hessian(dpos, step=0.1*utils.arcmin)
		hess  = 0.5*(hess+hess.T)
		try:
			pcov  = np.linalg.inv(0.5*hess)
			ddpos = np.diag(pcov)**0.5
			pcorr = pcov[0,1]/ddpos[0]/ddpos[1]
		except np.linalg.LinAlgError:
			ddpos = np.array([np.inf,np.inf])
			pcorr = 0
		return bunch.Bunch(dpos=dpos, ddpos=ddpos, pcorr=pcorr, poss_cel=np.array(poss_cel),
				poss_hor=np.array(poss_hor), amps=np.array(amps), damps=np.array(vamps)**0.5,
				models=models, nsrc=len(poss_cel))
	def find_starting_point(self):
		if True:
			poss, vals = [], []
			for i in range(self.nsrc):
				pos, val = self.liks[i].simple_max()
				poss.append(pos)
				vals.append(val)
			best = np.argmax(vals)
			dpos_cel = poss[best] - self.sdata[best].srcpos
			dpos = self.trfs[best].from_cel(dpos_cel)
			return dpos
		else:
			bpos, bval = None, np.inf
			for ddec in np.linspace(-self.rmax, self.rmax, self.ngrid):
				for dra in np.linspace(-self.rmax, self.rmax, self.ngrid):
					dpos  = np.array([dra,ddec])
					chisq = self.calc_chisq(dpos)
					if chisq < bval: bpos, bval = dpos, chisq
			return bpos
	def fit(self, verbose=False):
		self.verbose = verbose
		t1   = time.time()
		dpos = self.find_starting_point()
		dpos = optimize.fmin_powell(self.calc_chisq_wrapper, dpos/self.scale, disp=False)*self.scale
		res  = self.calc_full_result(dpos)
		res.time = time.time()-t1
		return res

def project_maps(imaps, pos, shape, wcs):
	pos   = np.asarray(pos)
	omaps = enmap.zeros((len(imaps),)+imaps[0].shape[:-2]+shape, wcs, imaps[0].dtype)
	pmap  = omaps.posmap()
	for i, imap in enumerate(imaps):
		omaps[i] = imaps[i].at(pmap+pos[i,::-1,None,None])
	return omaps

# Load source database
#srcpos = np.loadtxt(args.srclist, usecols=(args.rcol, args.dcol)).T*utils.degree
# We use ra,dec ordering in source positions here

# Set up shifted map geometry
shape, wcs = enmap.geometry(
		pos=np.array([[-1,-1],[1,1]])*args.orad*utils.arcmin, res=args.ores*utils.arcmin, proj="car")

# Set up fit output
f = open(args.odir + "/fit_rank_%03d.txt" % comm.rank, "w")

for ind in range(comm.rank, len(args.ifiles), comm.size):
	ifile = args.ifiles[ind]
	sdata = read_sdata(ifile)

	fitter = SrcFitter(sdata, fwhm)
	# Find the ML position
	t1     = time.time()
	fit    = fitter.fit(verbose=False)
	t2     = time.time()
	# Output summary
	for i in range(fit.nsrc):
		ostr = "%s %7.4f %7.4f %7.4f %7.4f %3d %7.4f %7.4f" % (sdata[i].id,
			fit.dpos[0]/utils.arcmin, fit.ddpos[0]/utils.arcmin,
			fit.dpos[1]/utils.arcmin, fit.ddpos[1]/utils.arcmin,
			sdata[i].sid, fit.amps[i]/1e3, fit.damps[i]/1e3)
		# Add some convenience data
		hour  = sdata[i].ctime/3600.%24
		ostr += " | %5.2f %9.4f %9.4f | %7.4f %2d" % (hour,
				fit.poss_hor[i,0]/utils.degree, fit.poss_hor[i,1]/utils.degree,
				(t2-t1)/fit.nsrc, fit.nsrc)
		print ostr
		f.write(ostr + "\n")
		f.flush()
	# Build shifted models
	smap = project_maps([s.map.preflat[0] for s in sdata], fit.poss_cel, shape, wcs)
	sdiv = project_maps([s.div for s in sdata], fit.poss_cel, shape, wcs)
	smod = project_maps(fit.models, fit.poss_cel, shape, wcs)
	smap = enmap.samewcs([smap,smod,smap-smod],smap)
	# And build scaled coadd
	trhs = np.sum(smap*sdiv*fit.amps[None,:,None,None],1)
	tdiv = np.sum(sdiv*fit.amps[:,None,None]**2,0)
	tdiv[tdiv==0] = np.inf
	tmap = trhs/tdiv
	# And write them
	enmap.write_map(args.odir + "/totmap_%s.fits" % sdata[0].id, tmap)
	for i in range(fit.nsrc):
		enmap.write_map(args.odir + "/shiftmap_%s_%03d.fits" % (sdata[i].id, sdata[i].sid), smap[:,i])
		# Output unshifted map too
		omap = enmap.samewcs([sdata[i].map.preflat[0],fit.models[i],sdata[i].map.preflat[0]-fit.models[i]],sdata[i].map)
		enmap.write_map(args.odir + "/fitmap_%s_%03d.fits" % (sdata[i].id,sdata[i].sid), omap)
f.close()
