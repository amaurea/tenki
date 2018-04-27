import numpy as np, os, h5py
from enlib import enmap, pmat, utils, scan, cg, bench, nmat, config, mpi, errors, array_ops
from enact import actdata, filedb, actscan
parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("area")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-n", "--nstep", type=int, default=50)
parser.add_argument("-m", "--method",type=str, default="messenger")
parser.add_argument(      "--ndet",  type=int, default=None)
parser.add_argument("-p", "--precompute", action="store_true")
parser.add_argument("-o", "--ostep", type=int, default=10)
args = parser.parse_args()

utils.mkdir(args.odir)
comm   = mpi.COMM_WORLD
dtype  = np.float32
ncomp  = 3
area   = enmap.read_map(args.area)
area   = enmap.zeros((ncomp,)+area.shape[-2:],area.wcs,dtype)
Tscale = 0.9
nstep  = args.nstep
downsample = config.get("downsample")

filedb.init()
ids   = filedb.scans[args.sel]
# Was 1e7
cooldown = sum([[10**j]*5 for j in range(6,0,-1)],[])+[1]

# Read my scans
njunk_tot = 0
cg_rhs    = area*0
cg_rjunk  = []
if args.precompute:
	prec_NNmap  = {lam: area*0 for lam in np.unique(cooldown)}
	prec_NNjunk = {lam: [] for lam in np.unique(cooldown)}
scans = []
for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	entry = filedb.data[id]
	try:
		scan  = actscan.ACTScan(entry)
		if scan.ndet == 0 or scan.nsamp == 0:
			raise errors.DataMissing("No samples in scan")
		if args.ndet:
			scan = scan[:args.ndet]
		if downsample > 1:
			scan = scan[:,::downsample]
		scan.pmap = pmat.PmatMap(scan, area)
		scan.pcut = pmat.PmatCut(scan)
		# Build the noise model
		tod = scan.get_samples()
		tod -= np.mean(tod,1)[:,None]
		tod  = tod.astype(dtype)
		scan.noise = scan.noise.update(tod, scan.srate)
		scan.T = np.min(scan.noise.D)*Tscale
		scan.noise_bar = nmat.NmatDetvecs(
			scan.noise.D-scan.T, scan.noise.V, scan.noise.E,
			scan.noise.bins, scan.noise.ebins, scan.noise.dets)
		# Set up cuts
		scan.cut_range = [njunk_tot,njunk_tot+scan.pcut.njunk]
		njunk_tot += scan.pcut.njunk
		# Prepare our filtered data. We do this one of two ways.
		# Either store Nb"d for each TOD, which can end up taking
		# up a lot of memory, or precompute P'(Nb"+(lT)")"Nbd"d for
		# each value of lambda. This saves memory if the maps aren't
		# too big and if the number of lambdas is reasonably small.
		# For 6 lambdas and deep56 size, we get 240 MB * 6 = 1.4 GB.
		# That corresponds to storing 4 downsampled tods.
		if args.precompute:
			iNbd = scan.noise_bar.apply(tod.copy())
			for lam in np.unique(cooldown):
				# Could cache this too, but it's fast to compute
				iNbt = nmat.NmatDetvecs(
						scan.noise_bar.iD + 1/(lam*scan.T), scan.noise_bar.iV,
						scan.noise_bar.iE, scan.noise_bar.bins,
						scan.noise_bar.ebins, scan.noise_bar.dets)
				work = iNbt.apply(iNbd.copy())
				work/= scan.T
				pjunk= np.zeros(scan.pcut.njunk, dtype)
				scan.pcut.backward(work, pjunk)
				scan.pmap.backward(work, prec_NNmap[lam])
				prec_NNjunk[lam].append(pjunk)
			del iNbd
		else:
			# Compute Nbd, which we need to store
			scan.Nbd = scan.noise_bar.apply(tod.copy())
		if args.method == "cg":
			scan.noise.apply(tod)
			tmp = np.zeros(scan.pcut.njunk,dtype)
			scan.pcut.backward(tod, tmp)
			scan.pmap.backward(tod, cg_rhs)
			cg_rjunk.append(tmp)
	except errors.DataMissing as e:
		print "Skipping %s (%s)" % (id, e.message)
		continue
	print "Read %s" % id
	scans.append(scan)

if args.precompute:
	for lam in prec_NNjunk:
		prec_NNmap[lam]  = utils.allreduce(prec_NNmap[lam], comm)
		prec_NNjunk[lam] = np.concatenate(prec_NNjunk[lam])

if args.method == "cg":
	cg_rhs = utils.allreduce(cg_rhs, comm)
	cg_rjunk = np.concatenate(cg_rjunk)
	if comm.rank == 0:
		enmap.write_map(args.odir + "/map_rhs.fits", cg_rhs)
	with h5py.File(args.odir + "/cut_rhs_%02d.hdf" % comm.rank, "w") as hfile:
		hfile["data"] = cg_rjunk

# Build div, which we need in both cases
div  = enmap.zeros((ncomp,)+area.shape,area.wcs,dtype)
for i in range(ncomp):
	work    = div[0]*0
	work[i] = 1
	for scan in scans:
		tod = np.zeros((scan.ndet,scan.nsamp),dtype)
		scan.pmap.forward(tod,  work)
		if args.method == "cg":
			scan.noise.white(tod)
		else: tod /= scan.T
		scan.pcut.backward(tod, np.zeros(scan.pcut.njunk,dtype))
		scan.pmap.backward(tod, div[i])
div = utils.allreduce(div, comm)
#idiv = utils.eigpow(div,-1,[0,1])
idiv = array_ops.svdpow(div,-1,[0,1], lim=1e-6)
if comm.rank == 0:
	enmap.write_map(args.odir + "/map_div.fits",  div)
	enmap.write_map(args.odir + "/map_idiv.fits", idiv)
del work, div
# And the same for junk
jdiv = np.full(njunk_tot, 1.0, dtype)
for scan in scans:
	tod = np.zeros((scan.ndet,scan.nsamp),dtype)
	scan.pcut.forward(tod, jdiv[scan.cut_range[0]:scan.cut_range[1]])
	if args.method == "cg":
		scan.noise.white(tod)
	else: tod /= scan.T
	scan.pcut.backward(tod, jdiv[scan.cut_range[0]:scan.cut_range[1]])
del tod

if args.method == "cg":
	with h5py.File(args.odir + "/cut_div_%02d.hdf" % comm.rank, "w") as hfile:
		hfile["data"] = jdiv

if args.method == "messenger":
	print cooldown
	#cooldown = [1e8]*1 + [1e6]*2 + [1e5]*5 + [1e4]*7 + [1e3]*7 + [1e2]*10 + [1e1] * 10

	map  = area*0
	junk = np.zeros(njunk_tot, dtype)
	plam = 0
	for i in range(nstep):
		lam = cooldown[i] if i < len(cooldown) else 1
		#lam = max(1,10**(6-i*0.4))
		for scan in scans:
			# Precompute Nb"+(lT)". Nb" = iD + iV iE iV'. since
			# T is diagonal, we can just add it to iD directly.
			# Could there be a fourier space unit issue, though?
			scan.iNbT = nmat.NmatDetvecs(
					scan.noise_bar.iD + 1/(lam*scan.T), scan.noise_bar.iV,
					scan.noise_bar.iE, scan.noise_bar.bins,
					scan.noise_bar.ebins, scan.noise_bar.dets)
		plam = lam
		# solve for t. We only use t[si] once, so we don't
		# actually need to store it separately like I do here.
		rhs = area*0
		for si, scan in enumerate(scans):
			t = np.zeros([scan.ndet,scan.nsamp],dtype)
			scan.pmap.forward(t, (lam*scan.T)**-1*map)
			scan.pcut.forward(t, (lam*scan.T)**-1*junk[scan.cut_range[0]:scan.cut_range[1]])
			if not args.precompute:
				t += scan.Nbd
			t  = scan.iNbT.apply(t)
			t /= scan.T
			scan.pcut.backward(t, junk[scan.cut_range[0]:scan.cut_range[1]])
			scan.pmap.backward(t, rhs)
		rhs    = utils.allreduce(rhs, comm)
		if args.precompute:
			rhs  += prec_NNmap[lam]
			junk += prec_NNjunk[lam]
		junk  /= jdiv
		map[:] = enmap.map_mul(idiv, rhs)
		if comm.rank == 0:
			print "%4d %15.7e %8.1f" % (i+1, np.std(map), lam)
			if (i+1) % args.ostep == 0:
				enmap.write_map(args.odir + "/map%04d.fits" % (i+1), map)

elif args.method == "cg":
	def A(x):
		map  = x[:area.size].reshape(area.shape)
		junk = x[area.size:]
		omap = map*0
		ojunk= junk*0
		for scan in scans:
			tod = np.zeros([scan.ndet,scan.nsamp],dtype)
			scan.pmap.forward(tod, map)
			scan.pcut.forward(tod, junk[scan.cut_range[0]:scan.cut_range[1]])
			scan.noise.apply(tod)
			scan.pcut.backward(tod, ojunk[scan.cut_range[0]:scan.cut_range[1]])
			scan.pmap.backward(tod, omap)
			del tod
		omap = utils.allreduce(omap, comm)
		return np.concatenate([omap.reshape(-1),ojunk],0)
	def M(x):
		map  = x[:area.size].reshape(area.shape)
		junk = x[area.size:]
		omap = map*0
		omap[:] = enmap.map_mul(idiv, map)
		ojunk= junk/jdiv
		return np.concatenate([omap.reshape(-1),ojunk],0)
	def dot(x,y):
		mprod = np.sum(x[:area.size]*y[:area.size])
		jprod = np.sum(x[area.size:]*y[area.size:])
		return mprod + comm.allreduce(jprod)

	bin = enmap.map_mul(idiv, cg_rhs)
	enmap.write_map(args.odir + "/map_bin.fits", bin)

	b = np.concatenate([cg_rhs.reshape(-1),cg_rjunk],0)
	solver = cg.CG(A, b, M=M, dot=dot)
	for i in range(nstep):
		solver.step()
		if comm.rank == 0:
			print "%5d %15.7e" % (solver.i, solver.err)
			if solver.i % args.ostep == 0:
				map = enmap.samewcs(solver.x[:area.size].reshape(area.shape),area)
				enmap.write_map(args.odir + "/map%04d.fits" % solver.i, map)
