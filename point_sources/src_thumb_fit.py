import numpy as np, os, time, h5py, astropy.io.fits, sys, argparse
from scipy import optimize
from enlib import utils, mpi, fft, enmap, bunch
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
parser.add_argument("-b", "--fwhm",  type=float, default=1.3)
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
	return sdata

# Single-source likelihood evaluator
class Srclik:
	def __init__(self, map, div, fwhm):
		self.map   = map.preflat[0]
		self.div   = div
		self.posmap= map.posmap()
		self.fwhm  = fwhm
		self.sigma = fwhm/(8*np.log(2))**0.5
	def calc_profile(self, pos):
		dpos = self.posmap - pos[:,None,None]
		dpos[1] *= np.cos(pos[0])
		r2   = np.sum(dpos**2,0)
		return np.exp(-0.5*r2/self.sigma**2)
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
		self.liks  = [Srclik(s.map, s.div, fwhm) for s in sdata]
		self.i     = 0
	def calc_chisq(self, x):
		dpos   = x*utils.arcmin
		chisqs = [lik.calc_chisq(s.srcpos+dpos) for lik,s in zip(self.liks,self.sdata)]
		chisq  = np.mean(chisqs)
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
	def fit(self):
		dpos = np.zeros(2)
		dpos = optimize.fmin_powell(self.calc_chisq, dpos/utils.arcmin, disp=False)*utils.arcmin
		res  = self.calc_full_model(dpos)
		return res

# Load source database
#srcpos = np.loadtxt(args.srclist, usecols=(args.rcol, args.dcol)).T*utils.degree

for ind in range(comm.rank, len(args.ifiles), comm.size):
	ifile = args.ifiles[ind]
	sdata = read_sdata(ifile)

	fitter = SrcFitter(sdata, fwhm)
	# Find the ML position
	fit    = fitter.fit()
	# Output map,model,resid for each




	print "dpos", dpos/utils.arcmin
