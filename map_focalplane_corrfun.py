# Map out the 2d correlation function of the focalplane, assuming it's
# stationary. Se average_cov.py if you want the full ndet*ndet covariance,
# which does not have this assumption.

import numpy as np, os, multiprocessing
from enlib import utils, mpi, config, errors, fft, array_ops, scanutils, enmap
from enact import filedb, actdata
parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("sel")
parser.add_argument("template")
parser.add_argument("ofile")
parser.add_argument("--nbin",  type=int,   default=100)
parser.add_argument("--fmax",  type=float, default=10)
parser.add_argument("--ntod",  type=int,   default=0)
parser.add_argument("--nmulti",type=int,   default=1)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

filedb.init()
comm  = mpi.COMM_WORLD
dtype = np.float32
nbin  = args.nbin
fmax  = args.fmax

# Read our template
template = enmap.read_map(args.template)

# Group into ar1+ar2+... groups
ids = filedb.scans[args.sel]
groups = scanutils.get_tod_groups(ids)
if args.ntod: groups = groups[:args.ntod]

# Our output array
rhs = enmap.zeros((nbin,)+template.shape[-2:], template.wcs, dtype)
div = rhs*0

def project_mat(pix, template, mat=None):
	if mat is None: mat = np.full([pix.shape[-1],pix.shape[-1]],1.0)
	pix  = np.asarray(pix)
	off  = np.asarray(template.shape[-2:])/2
	rpix = (pix[:,:,None] - pix[:,None,:]) + off[:,None,None]
	# Flatten
	rpix = rpix.reshape(2,-1)
	mat  = np.asarray(mat).reshape(-1)
	res  = utils.bin_multi(rpix, template.shape[-2:], weights=mat)
	return enmap.samewcs(res, template)

# Loop through and analyse each tod-group
for ind in range(comm.rank, len(groups), comm.size):
	ids     = groups[ind]
	entries = [filedb.data[id] for id in ids]
	try:
		d = actdata.read(entries, verbose=args.verbose)
		d = actdata.calibrate(d,  verbose=args.verbose, exclude=["autocut"])
		if d.ndet < 2 or d.nsamp < 2: raise errors.DataMissing("No data in tod")
	except (errors.DataMissing, AssertionError, IndexError) as e:
		print "Skipping %s (%s)" % (str(ids), e.message)
		continue
	print "Processing %s" % (str(ids))
	tod  = d.tod
	del d.tod
	tod -= np.mean(tod,1)[:,None]
	tod  = tod.astype(dtype)
	ft   = fft.rfft(tod)
	nfreq= ft.shape[-1]
	dfreq= d.srate/d.nsamp
	del tod

	# Get the pixel position of each detector in order [{y,x},ndet]
	pix  = np.round(template.sky2pix(d.point_template.T[::-1])).astype(int)

	# This operation is slow. We can't use openmp because python is stupid,
	# and it isn't worth it to put this in fortran. So use multiporcessing
	def handle_bin(fbin):
		print fbin
		f1,f2 = [min(nfreq-1,int(i*fmax/dfreq/nbin)) for i in [fbin,fbin+1]]
		fsub  = ft[:,f1:f2]
		cov   = array_ops.measure_cov(fsub)
		std   = np.diag(cov)**0.5
		corr  = cov / std[:,None] / std[None,:]
		myrhs = project_mat(pix, template, corr)
		mydiv = project_mat(pix, template)
		return fbin, myrhs, mydiv
	def collect(args):
		fbin, myrhs, mydiv = args
		rhs[fbin] += myrhs
		div[fbin] += mydiv
	p = multiprocessing.Pool(args.nmulti)
	for fbin in range(nbin):
		p.apply_async(handle_bin, [fbin], callback=collect)
	p.close()
	p.join()
	del ft

# Collect the results
if comm.rank == 0: print "Reducing"
rhs = enmap.samewcs(utils.allreduce(rhs, comm), rhs)
div = enmap.samewcs(utils.allreduce(div, comm), div)
with utils.nowarn():
	map = rhs/div

if comm.rank == 0:
	print "Writing"
	enmap.write_map(args.ofile, map)
