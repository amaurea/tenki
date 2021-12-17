# Program to analyse the results of forced_photometry_subarray.py.
# Inputs: a set of files covering a time period where most sources
# are assumed to be constant. These will be used to calibrate
# detector groups. After calibration, the light-curve of a given
# source will be computed using the detector groups.

import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("targets")
parser.add_argument("ifiles", nargs="+")
parser.add_argument("ofile")
parser.add_argument("-C", "--calib",  type=int,   default=0)
parser.add_argument("-n", "--ngroup", type=int,   default=4)
parser.add_argument("-s", "--minsn",  type=float, default=None)
args = parser.parse_args()
import numpy as np, h5py, emcee
from pixell import utils, bunch
from scipy import optimize

# Get the list of source indices we want to measure the light-curve for
targets = [int(w) for w in args.targets.split(",")]

# We want to analyze by frequency, but it's still useful to be able to
# keep track of the arrays. We want to be able to calibrate the arrays
# separately, but keep fluxes consistent between them. So let's group
# by frequency first, and then split into arrays later. We can recover
# the array from the detector ids later.
tags    = [os.path.basename(ifile).split(".")[2] for ifile in args.ifiles]
freqs   = [tag.split("_")[1] for tag in tags]
fgroups = utils.find_equal_groups(freqs)

def read_hdf(ifile):
	res = bunch.Bunch()
	with h5py.File(ifile, "r") as hfile:
		for key in hfile:
			res[key] = hfile[key][()]
	return res

def calc_dt(t, t0, ivar):
	return np.sum((t-t0[:,None])*ivar,0)/np.sum(ivar,0)

def solve(rhs, ivar):
	with utils.nowarn():
		res = rhs/ivar
		res[~np.isfinite(res)] = 0
	return res

def build_detector_groups(t, ivar, ngroup=4):
	"""Build groups of detectors based on when they observe"""
	tref = np.sum(t*ivar)/np.sum(ivar)
	with utils.nowarn():
		rel_ivar = 1/(1/ivar + 1/ivar[0])
	src_dt = np.sum((t-t[0])*rel_ivar, 1)/np.sum(rel_ivar,1)
	det_dt = np.sum((t-src_dt[:,None]-tref)*ivar,0)/np.sum(ivar,0)
	inds   = np.argsort(det_dt)
	ndet   = t.shape[1]
	groups = []
	for i in range(ngroup):
		groups.append(inds[i*ndet//ngroup:(i+1)*ndet//ngroup])
	return groups

def build_detector_groups_multiarray(t, ivar, agroups, ngroup=4):
	groups = []
	ainds  = []
	for ai, adets in enumerate(agroups):
		adets     = np.array(adets)
		subgroups = build_detector_groups(t[:,adets], ivar[:,adets], ngroup=ngroup)
		# Rewrite indices to refer to the full detector index instead of the index into
		# the sliced arrays
		subgroups = [adets[g] for g in subgroups]
		# And append to our results
		for group in subgroups:
			groups.append(group)
			ainds.append(ai)
	return groups, ainds

def infer_gain_emcee(gflux, givar, nsamp=1000):
	# Returns the relative gain for each group in gflux[nsrc,ngroup]
	# and their uncertainty. The relative gain is defined such that the
	# mean is zero, and one calibrates gflux by multiplying by these gains.
	ngroup  = gflux.shape[1]
	nwalker = ngroup*6
	def zip(gain): return gain[1:]
	def unzip(x): return np.concatenate([[1],x])
	def calc_sflux(gflux, givar, gain):
		gflux_norm = gflux*gain
		givar_norm = givar/gain**2
		sivar = np.sum(givar_norm,1)
		sflux = np.sum(gflux_norm*givar_norm,1)/sivar
		return sflux, sivar
	def calc_log_prob(x):
		# d = sg + n, n cov N; d = gflux, s = sflux, g = gain; givar = N"
		# First get lik in terms of sflux, so we can marginalize over it
		# -2logL = (d - sg)'N"(d - sg) + log |2piN|
		#        = (s - (g'N"g)"g'N"d)'g'N"g(s - ) + d'N"d - ŝ'S"ŝ + log |2piN|, ŝ = (g'N"g)"g'N"d, S" = g'N"g
		# Marginalize over s:
		# -2logL = -log |2piS| - ŝ'S"s + d'N"d + log |2piN|
		# d-term and N terms are constant, so ignore.
		# Ignoring the S det is equivalent to Jeffrey's prior
		# We are left with logL = 0.5*ŝ'S"ŝ.
		if np.any((x < 0.5)|(x>2)): return -1e6
		gain = unzip(x)
		sflux, sivar = calc_sflux(gflux, givar, gain)
		return 0.5*np.sum(sflux**2*sivar)
	x0      = zip((1 + np.random.standard_normal([nwalker,ngroup])*0.1).T).T
	sampler = emcee.EnsembleSampler(*x0.shape, calc_log_prob)
	sampler.run_mcmc(x0, nsamp)
	samples = sampler.get_chain(flat=True)
	gain    = unzip(np.mean(samples,0))
	dgain   = unzip(np.std (samples,0))
	dgain[0] = np.mean(dgain[1:])
	# We want a mean gain of 1
	gain   /= np.mean(gain)
	return gain, dgain

# Process files array by array
with open(args.ofile, "w") as ofile:
	for fi, inds in enumerate(fgroups):
		freq = freqs[inds[0]]
		# We assume that the files cover only a single pass over an area of the sky,
		# but that this could be split into a few files. That means that if two files
		# have the same detector seeing the same source, then that's just the same
		# observation that has been split. So we want to merge the data together.
		# The simplest structure to work with will be [nsrc,ndet,{frhs,trhs,ivar}].
		# This requires us to first find the full set of sources and detectors involved
		fdatas   = [read_hdf(args.ifiles[ind]) for ind in inds]
		dets = np.unique(np.concatenate([d.dets for d in fdatas]))
		sids = np.unique(np.concatenate([d.sids for d in fdatas]))
		ndet, nsrc = len(dets), len(sids)
		data     = np.zeros([nsrc,ndet,3])
		for d in fdatas:
			src_inds = utils.find(sids, d.sids)
			det_inds = utils.find(dets, d.dets)
			ivar     = d.dflux**-2
			tmp      = np.zeros([len(d.sids),ndet,3])
			tmp[:,det_inds,0] += d.flux * ivar
			tmp[:,det_inds,1] += d.t    * ivar
			tmp[:,det_inds,2] += ivar
			data[src_inds] += tmp
		# Find our targets and non-targets
		targ_inds = utils.find_any(sids, targets)
		if len(targ_inds) == 0:
			print("No targets found for %s. Skipping" % freq)
			continue
		# Prune detectors that don't hit any of our targets
		good = np.sum(data[targ_inds,:,2],0) > 0
		data = data[:,good]
		dets = dets[good]
		# At this point we have data from all the different arrays at this freq in one array.
		# Let's make a list of the detector indices for each array. We will use this to
		# make per-array groups later.
		det_arrs = np.char.partition(np.char.decode(dets), "_")[:,0]
		det_arr_groups = utils.find_equal_groups(det_arrs)
		arrs     = [det_arrs[g[0]] for g in det_arr_groups]
		# Build our reference set of sources
		rest_inds = set(range(nsrc)) - set(targ_inds)
		if args.minsn:
			sn = solve(np.sum(data[:,:,0],1), np.sum(data[:,:,2],1)**0.5)
			rest_inds -= set(np.where(sn < args.minsn)[0])
		rest_inds = sorted(list(rest_inds))
		print(rest_inds)
		#rest_inds = np.array(sorted(list(set(np.arange(nsrc))-set(targ_inds))))
		ntarg, nrest = len(targ_inds), len(rest_inds)
		frhs, trhs, ivar = np.moveaxis(data,2,0)
		t       = solve(trhs, ivar)
		dgroups, aind = build_detector_groups_multiarray(t[targ_inds], ivar[targ_inds], det_arr_groups, ngroup=args.ngroup)
		ngroup = len(dgroups)
		# Ok, now that we have our groups, measure the mean flux of each reference
		# source in each bin
		gflux, gt, givar = np.zeros([3, nsrc, ngroup])
		for gi, dgroup in enumerate(dgroups):
			givar[:,gi] = np.sum(ivar[:,dgroup],1)
			gflux[:,gi] = solve(np.sum(frhs[:,dgroup],1), givar[:,gi])
			gt[:,gi]    = solve(np.sum(trhs[:,dgroup],1), givar[:,gi])

		# The overall amplitude of each source is unknown, but is assumed to be
		# constant during this period. Hence we can model gflux as
		#  gflux[ngroup,nsrc] = sflux[nsrc] * gain[ngroup] + noise
		# Both sflux and gain are unknown. Will use a likelihood exploration to recover the gain
		if args.calib:
			gain, dgain = infer_gain_emcee(gflux[rest_inds], givar[rest_inds])
		else:
			gain, dgain = np.full(ngroup,1), np.full(ngroup,0)
		# Measure the light-curves for our target sources, and apply gain
		targ_flux  = gflux[targ_inds]       * gain
		targ_dflux = (givar[targ_inds]**-1  * gain**2 + targ_flux**2 * dgain**2)**0.5

		# Output results. We want to do this by array, so first group by array. I'm running out of
		# names for groups of things
		agroups = utils.find_equal_groups(aind)

		for ti in targ_inds:
			for ai, agroup in enumerate(agroups):
				arr = arrs[aind[agroup[0]]]
				for gi in agroup:
					desc = ("%s %s %5d %15.2f %8.3f %8.3f %8.3f %8.3f" % (
						arr, freq, sids[ti], gt[ti,gi], targ_flux[ti,gi], targ_dflux[ti,gi],
						gain[gi], dgain[gi]))
					print(desc)
					ofile.write(desc + "\n")
				print()
				ofile.write("\n")
				ofile.flush()
