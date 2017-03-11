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
		self.map   = map
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
		damp = 1/np.sum(profile*self.div*profile)
		amp  = damp*np.sum(profile*self.div*self.map)
		return amp, damp
	def calc_model(self, pos):
		profile  = self.calc_profile(pos)
		amp, damp= self.calc_amp(profile)
		return profile*amp
	def calc_chisq(self, posoff):
		model = self.calc_model(posoff)
		resid = self.map - model
		chisq = np.sum(resid**2)
		return chisq

class SrcFitter:
	def __init__(self, sdata, fwhm):
		self.nsrc  = len(sdata)
		self.sdata = sdata
		self.liks  = [Srclik(s.map, s.div, fwhm) for s in sdata]
		self.i     = 0
	def calc_chisq(self, dpos):
		chisqs = [lik.calc_chisq(s.srcpos+dpos) for lik,s in zip(self.liks,self.sdata)]
		chisq  = np.sum(chisqs)
		print "%4d %9.4f %9.4f %15.7e" % (self.i, dpos[0]/utils.arcmin, cpos[1]/utils.arcmin, chisq)
		self.i += 1
		return chisq
	def fit(self):
		dpos = np.zeros(self.nsrc)
		dpos = optimize.fmin_powell(self.calc_chisq, dpos, disp=False)
		self.i += 1
		return dpos

# Load source database
#srcpos = np.loadtxt(args.srclist, usecols=(args.rcol, args.dcol)).T*utils.degree

for ind in range(comm.rank, len(args.ifiles), comm.size):
	ifile = args.ifiles[ind]
	sdata = read_sdata(ifile)

	fitter = SrcFitter(sdata, fwhm)
	dpos   = fitter.fit()
