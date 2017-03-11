import numpy as np, os, time, h5py, astropy.io.fits, sys, argparse
from scipy import optimize
from enlib import utils, mpi, fft, enmap, bunch
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
parser.add_argument("-b", "--fwhm",  type=float, default=1.3)
parser.add_argument("-x", "--file-srcpos-decra", action="store_true")
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
				srcpos = g["srcpos"].value)
			if args.file_srcpos_decra:
				sdata[ind].srcpos = sdata[ind].srcpos[::-1]
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
		#enmap.write_map("rvec.fits",  rvec)
		#enmap.write_map("rpara.fits", rpara)
		#enmap.write_map("rorto.fits", rorto)
		# Evaluate each beam component
		ipara    = rpara/self.res+self.vbeam.size/2
		bpara    = utils.interpol(self.vbeam, ipara[None], mask_nan=False, order=self.order, prefilter=False)
		borto    = np.exp(-0.5*rorto**2/self.sigma**2)
		res      = enmap.samewcs(bpara*borto, rvec)
		#enmap.write_map("test.fits", res)
		#enmap.write_map("testA.fits", enmap.samewcs(bpara,rvec))
		#enmap.write_map("testB.fits", enmap.samewcs(borto,rvec))
		#1/0
		return res

# Single-source likelihood evaluator
class Srclik:
	def __init__(self, map, div, beam, maxdist=8):
		self.map   = map.preflat[0]
		#enmap.write_map("map.fits", map)
		self.div   = div
		self.posmap= map.posmap()
		#enmap.write_map("posmap.fits", self.posmap)
		self.beam  = beam
	def calc_profile(self, pos):
		dpos = self.posmap[::-1] - pos[:,None,None]
		return self.beam.eval(dpos)
	def calc_amp(self, profile):
		vamp = 1/np.sum(profile*self.div*profile)
		amp  = vamp*np.sum(profile*self.div*self.map)
		if ~np.isfinite(amp): amp = 0
		return amp, vamp
	def calc_model(self, pos):
		profile  = self.calc_profile(pos)
		amp, vamp= self.calc_amp(profile)
		return profile*np.abs(amp)
	def calc_chisq(self, posoff):
		model = self.calc_model(posoff)
		resid = self.map - model
		chisq = np.mean(resid**2)
		return chisq

class SrcFitter:
	def __init__(self, sdata, fwhm):
		self.nsrc  = len(sdata)
		self.scale = utils.arcmin
		self.sdata = sdata
		self.liks  = []
		for s in sdata:
			beam = BeamModel(fwhm, s.vel, s.srcpos[1], s.fknee, s.alpha)
			lik  = Srclik(s.map, s.div, beam)
			self.liks.append(lik)
		self.i     = 0
		self.verbose = False
	def calc_chisq(self, x):
		dpos   = x*utils.arcmin
		chisqs = [lik.calc_chisq(s.srcpos+dpos) for lik,s in zip(self.liks,self.sdata)]
		chisq  = np.mean(chisqs)
		if self.verbose:
			print "%4d %9.4f %9.4f %15.7e" % (self.i, dpos[0]/utils.arcmin, dpos[1]/utils.arcmin, chisq)
		self.i += 1
		return chisq
	def calc_full_model(self, dpos):
		amps, models, poss = [], [], []
		for i in range(self.nsrc):
			lik, sd = self.liks[i], self.sdata[i]
			profile = lik.calc_profile(sd.srcpos+dpos)
			amp, vamp = lik.calc_amp(profile)
			amps.append(amp)
			models.append(amp*profile)
			poss.append(sd.srcpos+dpos)
		return bunch.Bunch(dpos=dpos, poss=np.array(poss), amps=np.array(amps),
				models=models, nsrc=len(poss))
	def fit(self, verbose=False):
		self.verbose = verbose
		dpos = np.zeros(2)
		dpos = optimize.fmin_powell(self.calc_chisq, dpos/utils.arcmin, disp=False)*utils.arcmin
		res  = self.calc_full_model(dpos)
		return res

# Load source database
#srcpos = np.loadtxt(args.srclist, usecols=(args.rcol, args.dcol)).T*utils.degree
# We use ra,dec ordering in source positions here

for ind in range(comm.rank, len(args.ifiles), comm.size):
	ifile = args.ifiles[ind]
	sdata = read_sdata(ifile)

	fitter = SrcFitter(sdata, fwhm)
	# Find the ML position
	fit    = fitter.fit()
	# Output summary
	print "%s %7.3f %7.3f" % (sdata[0].id, fit.dpos[0]/utils.arcmin, fit.dpos[1]/utils.arcmin),
	for i in range(fit.nsrc):
		print " |%3d %7.4f" % (sdata[i].sid, fit.amps[i]/1e3),
	print
	# Output map,model,resid for each
	for i in range(fit.nsrc):
		omap = enmap.samewcs([sdata[i].map.preflat[0],fit.models[i],sdata[i].map.preflat[0]-fit.models[i]],sdata[i].map)
		enmap.write_map(args.odir + "/fitmap_%s_%03d.fits" % (sdata[i].id,sdata[i].sid), omap)
