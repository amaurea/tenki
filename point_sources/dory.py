# This program is a simple alternative to nemo. It uses the same
# constant-covariance noise model which should make it fast but
# suboptimal. I wrote this because I was having too much trouble
# with nemo. The aim is to be fast and simple to implement.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("idiv")
parser.add_argument("odir")
parser.add_argument("-m", "--mask",    type=str,   default=None)
parser.add_argument("-b", "--beam",    type=str,   default="1.4")
parser.add_argument("-R", "--regions", type=str,   default=None)
parser.add_argument("-a", "--apod",    type=int,   default=30)
parser.add_argument("-s", "--nsigma",  type=float, default=3.5)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()
import numpy as np
from scipy import ndimage
from enlib import enmap, utils, bunch, mpi, fft

def get_beam(fname):
	try:
		sigma = float(fname)*utils.arcmin*utils.fwhm
		l     = np.arange(40e3)
		beam  = np.exp(-0.5*(l*sigma)**2)
	except ValueError:
		beam = np.loadtxt(fname, usecols=(1,))
	return beam

def get_regions(fname, shape, wcs):
	# Set up our regions. No significant automation for now.
	if fname:
		# Region file has format ra1 ra2 dec1 dec2. Make it
		# [:,{from,to},{dec,ra}] so it's compatible with enmap bounding boxes
		regions  = np.loadtxt(fname)[:,:4]
		regions  = np.transpose(regions.reshape(-1,2,2),(0,2,1))[:,:,::-1]
		regions *= utils.degree
		# And turn them into pixel bounding boxes
		regions = np.array([enmap.skybox2pixbox(shape, wcs, box) for box in regions])
		regions = np.round(regions).astype(int)
	else:
		regions = np.array([[0,0],shape[-2:]])[None]
	return regions

def get_apod_holes(div, pixrad):
	return enmap.samewcs(0.5*(1-np.cos(np.pi*np.minimum(1,ndimage.distance_transform_edt(div>0)/float(pixrad)))))

def smooth_ps_gauss(ps, lsigma):
	"""Smooth a 2d power spectrum to the target resolution in l. Simple
	gaussian smoothing avoids ringing."""
	# First get our pixel size in l
	ly, lx = enmap.laxes(ps.shape, ps.wcs)
	ires   = np.array([ly[1],lx[1]])
	sigma_pix = np.abs(lsigma/ires)
	fmap  = enmap.fft(ps)
	ky    = np.fft.fftfreq(ps.shape[-2])*sigma_pix[0]
	kx    = np.fft.fftfreq(ps.shape[-1])*sigma_pix[1]
	kr2   = ky[:,None]**2+kx[None,:]**2
	fmap *= np.exp(-0.5*kr2)
	return enmap.ifft(fmap).real

def safe_mean(arr, bsize=100):
	arr   = arr.reshape(-1)
	nblock = arr.size//bsize
	if nblock <= 1: return np.mean(arr)
	means = np.mean(arr[:nblock*bsize].reshape(nblock,bsize),-1)
	means = np.concatenate([means,[np.mean(arr[(nblock-1)*bsize:])]])
	return np.median(means)

def get_snmap_norm(snmap, bsize=240):
	norm = snmap*0+1
	ny, nx = np.array(snmap.shape[-2:])//bsize
	for by in range(ny):
		y1 = by*bsize
		y2 = (by+1)*bsize if by < ny-1 else snmap.shape[-2]
		for bx in range(nx):
			x1 = bx*bsize
			x2 = (bx+1)*bsize if bx < nx-1 else snmap.shape[-1]
			sub  = snmap[y1:y2,x1:x2]
			vals = sub[sub!=0]
			if vals.size == 0: continue
			std  = safe_mean(vals**2)**0.5
			norm[y1:y2,x1:x2] = std
	return norm

def measure_noise(noise_map, margin=15, apod=15, ps_res=200):
	# Ignore the margin and apodize the rest, while keeping the same overall shape
	apod_map  = enmap.extract((noise_map[margin:-margin,margin:-margin]*0+1).apod(apod), noise_map.shape, noise_map.wcs)
	noise_map = noise_map*apod_map
	ps        = np.abs(enmap.fft(noise_map))**2
	# Normalize to account for the masking
	ps /= np.mean(apod_map**2)
	#enmap.write_map("ps1.fits", ps*0+np.fft.fftshift(ps))
	ps     = smooth_ps_gauss(ps, ps_res)
	#enmap.write_map("ps2.fits", ps*0+np.fft.fftshift(ps))
	return ps

def build_filter(ps, beam):
	# Build our matched filter, assumping beam-shaped point sources
	lmap   = ps.modlmap()
	beam2d = np.interp(lmap, np.arange(len(beam)), beam)
	filter = beam2d/ps
	# Construct the 
	m  = enmap.ifft(beam2d+0j).real
	m /= m[0,0]
	norm = enmap.ifft(enmap.fft(m)*filter).real[0,0]
	filter /= norm
	return filter, beam2d

def get_thumb(map, size):
	return enmap.shift(map, (size//2, size//2))[:size,:size]

def get_template(filter, beam2d, size):
	# Build the real-space template representing the
	# response of the filter to a unit-amplitude point source
	template  = enmap.ifft(filter*beam2d+0j).real
	template /= np.max(template)
	template  = get_thumb(template, size=size)
	return template

def fit_srcs(fmap, labels, inds, extended_threshold=1.1):
	# Our normal fit is based on the center of mass. This is
	# probably a bit suboptimal for faint sources, but those will
	# be pretty bad anyway.
	pos_com = np.array(ndimage.center_of_mass(fmap, labels, inds))
	amp_com = fmap.at(pos_com.T, unit="pix")
	# We compare these amplitudes with the maxima. Normally these
	# will be very close. If they are significantly different, then
	# this is probably an extended object. To allow the description
	# of these objects as a sum of sources, it's most robust to use
	# the maximum positions and amplitudes here.
	pos_max = np.array(ndimage.maximum_position(fmap, labels, inds))
	amp_max = np.array(ndimage.maximum(fmap, labels, inds))
	pos, amp = pos_com.copy(), amp_com.copy()
	extended = amp_max > amp_com*extended_threshold
	pos[extended] = pos_max[extended]
	amp[extended] = amp_max[extended]
	return pos, amp

def calc_model(shape, wcs, ipos, amps, template):
	model = enmap.zeros(shape, wcs, template.dtype)
	size  = np.array(template.shape)
	dbox  = np.array([[0,0],size])-size//2
	for i, pix in enumerate(ipos):
		pix0     = utils.nint(pix)
		srcmodel = fft.shift(template, pix-pix0)*amps[i]
		enmap.insert_at(model, pix0+dbox, srcmodel, op=lambda a,b:a+b, wrap=shape[-2:])
	return model

def sim_initial_noise(div, lknee=3000, alpha=-2):
	# Simulate white noise
	noise = enmap.rand_gauss(div.shape, div.wcs, div.dtype)
	l     = div.modlmap()
	profile = 1 + ((l+0.5)/lknee)**alpha
	profile[0,0] = 0
	noise  = enmap.ifft(enmap.fft(noise)*profile).real
	noise[div>0] *= div[div>0]**-0.5
	return noise

def find_srcs(imap, idiv, beam, apod=15, snmin=3.5, npass=2, snblock=5, nblock=10,
		ps_res=2000, pixwin=True, kernel=256, dump=None, verbose=False):
	if dump: utils.mkdir(dump)
	# Apodize a bit before any fourier space operations
	apod_map = (idiv*0+1).apod(apod) * get_apod_holes(idiv,apod)
	imap = imap*apod_map
	# Deconvolve the pixel window from the beginning, so we don't have to worry about it
	if pixwin: imap = enmap.apply_window(imap,-1)
	# Whiten the map
	wmap   = imap * idiv**0.5
	adiv   = idiv * apod_map**2
	#print "max(imap)", np.max(imap)
	#print "median(adiv)**-0.5", np.median(adiv)**-0.5
	#print "max(wmap)", np.max(wmap), np.max(imap)/np.median(adiv)**-0.5
	noise  = sim_initial_noise(idiv)
	edge_mask = None
	for ipass in range(npass):
		wnoise = noise * adiv**0.5
		# From now on we treat the whitened map as the real one. And assume that
		# we only need a constant covariance model. If div has lots of structure
		# on the scale of the signal we're looking for, then this could introduce
		# false detections. Empirically this hasn't been a problem, though.
		ps             = measure_noise(wnoise, apod, apod, ps_res=ps_res)
		filter, beam2d = build_filter(ps, beam)
		template       = get_template(filter, beam2d, size=kernel)
		fmap           = enmap.ifft(filter*enmap.fft(wmap)).real
		fnoise         = enmap.ifft(filter*enmap.fft(wnoise)).real
		norm           = get_snmap_norm(fnoise*(apod_map==1))
		del wnoise
		if dump:
			enmap.write_map(dump + "/wnoise_%02d.fits" % ipass, wnoise)
			enmap.write_map(dump + "/wmap_%02d.fits"   % ipass, wmap)
			enmap.write_map(dump + "/fmap_%02d.fits"   % ipass, fmap)
			enmap.write_map(dump + "/norm_%02d.fits"   % ipass, norm)
		result = bunch.Bunch(snmap=fmap/norm)
		fits   = bunch.Bunch(amps=[], damps=[], pix=[], npix=[])
		for iblock in range(nblock):
			snmap  = fmap/norm
			if dump:
				wnmap.write_map(dump + "/snmap_%02d_%02d.fits" % (ipass, iblock), snmap)
			# Find and mask sources at the edge of the apodization region.
			# This doen't need a very accurate filter, so it can be done once
			# and for all the first time.
			if edge_mask is None:
				matches   = (snmap >= snmin)|(apod_map < 1)
				labels, nlabel  = ndimage.label(matches)
				edge_mask = labels == labels[0,0]
			# Find all sufficiently strong candidates
			matches   = snmap >= snmin
			labels, nlabel  = ndimage.label(matches)
			if nlabel == 0: break
			all_inds = np.arange(nlabel)
			sn       = ndimage.maximum(snmap,     labels, all_inds+1)
			on_edge  = ndimage.maximum(edge_mask, labels, all_inds+1)
			# Strong sources cause ringing that can mask nearby sources and/or secondary matches.
			# We will therefore keep only the strongest matches, subtract them from snmap,
			# and then repeat.
			sn_lim = np.max(sn[~on_edge], initial=0)
			keep   = np.where((sn>=snmin)&(sn >= sn_lim/snblock))[0]
			if len(keep) == 0: break
			# This is a bit lazy. It can end up using just 1 pixel for the pos for weak srcs
			pix, amps = fit_srcs(fmap, labels, keep+1)
			damps      = norm.at(pix.T, unit="pix", order=0)
			#print "sns", amps/damps
			npix       = ndimage.sum(matches, labels, keep+1)
			model      = calc_model(fmap.shape, fmap.wcs, pix, amps, template)
			fmap      -= model
			fits.amps.append(amps)
			fits.damps.append(damps)
			fits.pix.append(pix)
			fits.npix.append(npix)
			if verbose:
				edges = [0,5,10,20,50,100,np.inf]
				sns   = np.concatenate(fits.amps)/np.concatenate(fits.damps)
				counts= np.histogram(sns, edges)[0]
				desc  = " ".join(["%d: %5d" % (e,c) for e,c in zip(edges[:-1],counts)])
				print "pass %d block %2d sn: %s" % (ipass+1, iblock+1, desc)
		# Concatenate passes, sort them and move them into result structure
		for name in fits: fits[name] = np.concatenate(fits[name])
		order = np.argsort(fits.amps/fits.damps)[::-1]
		for name in fits: result[name] = fits[name][order]
		del fits
		# Rescale amplitudes to compensate for the initial whitening
		rms        = adiv.at(result.pix.T, unit="pix", order=0)**-0.5
		result.amps  *= rms
		result.damps *= rms
		# Get physical coordinates
		result.pos    = wmap.pix2sky(result.pix.T).T
		# Compute model and residual in real units
		result.resid_snmap = fmap/norm
		beam_thumb  = get_thumb(enmap.ifft(beam2d+0j).real, size=kernel)
		beam_thumb /= np.max(beam_thumb)
		result.model       = calc_model(imap.shape, imap.wcs, result.pix, result.amps, beam_thumb)
		result.resid       = imap - result.model
		result.map         = imap
		noise = result.resid
	return result

def write_catalog(ofile, result):
	np.savetxt(ofile, np.array([
		result.pos[:,1]/utils.degree,
		result.pos[:,0]/utils.degree,
		result.amps/result.damps,
		result.amps/1e3,
		result.damps/1e3,
		result.npix,
	]).T, fmt="%9.4f %9.4f %8.3f %9.4f %9.4f %5d")

comm       = mpi.COMM_WORLD
shape, wcs = enmap.read_map_geometry(args.imap)
beam       = get_beam(args.beam)
regions    = get_regions(args.regions, shape, wcs)
ps_res     = 200
utils.mkdir

# Process each region. Will do mpi here for now. This is a bit lazy - it
# completely skips any mpi speedups for single-region maps. Everything is
# T-only for now too
for ri in range(comm.rank, len(regions), comm.size):
	region = regions[ri]
	imap   = enmap.read_map(args.imap, pixbox=region).preflat[0]
	idiv   = enmap.read_map(args.idiv, pixbox=region).preflat[0]
	if args.mask: idiv *= enmap.read_map(args.mask, pixbox=region).preflat[0]
	result = find_srcs(imap, idiv, beam, apod=args.apod, snmin=args.nsigma, verbose=args.verbose)
	# Write region output
	prefix = args.odir + "/region_%02d_" % ri
	write_catalog(prefix + "cat.txt" , result)
	for name in ["map","snmap","model","resid","resid_snmap"]:
		enmap.write_map(prefix + name + ".fits", result[name])
