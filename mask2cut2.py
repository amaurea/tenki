from __future__ import division, print_function
import numpy as np, os
from enlib import utils
with utils.nowarn(): import h5py
from enlib import config, sampcut, mpi, pmat, scan as enscan, enmap
from enlib import coordinates, errors, flagrange, bunch, memory
from enact import filedb, actdata, files

config.default("pmat_accuracy", 100.0, "Factor by which to lower accuracy requirement in pointing interpolation. 1.0 corresponds to 1e-3 pixels and 0.1 arc minute in polangle")

parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("sel")
parser.add_argument("objname")
parser.add_argument("mask")
parser.add_argument("odir")
parser.add_argument("--margin", type=float, default=1)
args = parser.parse_args()

filedb.init()
ids  = filedb.scans[args.sel]
comm = mpi.COMM_WORLD
dtype= np.float32
ntod = len(ids)

margin = args.margin*utils.degree

imask = enmap.read_map(args.mask)
# Expand to 3 components, as the pointing code expects that
mask = enmap.zeros((3,)+imask.shape[-2:], imask.wcs, dtype)
mask[0] = imask.reshape((-1,)+imask.shape[-2:])[0]
del imask

utils.mkdir(args.odir)

# Each mpi tasks opens its own work file
wfname  = args.odir + "/work_%03d.hdf" % comm.rank
mystats = []
with h5py.File(wfname, "w") as hfile:
	for ind in range(comm.rank*ntod//comm.size, (comm.rank+1)*ntod//comm.size):
		id    = ids[ind]
		entry = filedb.data[id]
		# We only need pointing to build this cut
		try:
			d = actdata.read(entry, ["point_offsets","boresight","site","array_info"])
			d = actdata.calibrate(d, exclude=["autocut"])
		except (errors.DataMissing, AttributeError) as e:
			print("Skipping %s (%s)" % (id, e))
			continue
		# Build a projector between samples and mask. This
		# requires us to massage d into scan form. It's getting
		# annoying that scan and data objects aren't compatible.
		bore = d.boresight.T.copy()
		bore[:,0] -= bore[0,0]
		scan = enscan.Scan(
			boresight = bore,
			offsets = np.concatenate([np.zeros(d.ndet)[:,None],d.point_offset],1),
			comps = np.concatenate([np.ones(d.ndet)[:,None],np.zeros((d.ndet,3))],1),
			mjd0 = utils.ctime2mjd(d.boresight[0,0]),
			sys = "hor", site = d.site)
		scan.hwp_phase = np.zeros([len(bore),2])
		bore_box = np.array([np.min(d.boresight,1),np.max(d.boresight,1)])
		bore_corners = utils.box2corners(bore_box)
		scan.entry = d.entry
		# Is the source above the horizon? If not, it doesn't matter how close
		# it is.
		mjd = utils.ctime2mjd(utils.mjd2ctime(scan.mjd0)+scan.boresight[::100,0])
		try:
			object_pos = coordinates.interpol_pos("cel","hor", args.objname, mjd, site=scan.site)
		except AttributeError as e:
			print("Unexpected error in interpol_pos for %s. mid time was %.5f. message: %s. skipping" % (id, mjd[len(mjd)//2], e))
			continue
		visible = np.any(object_pos[1] >= -margin)
		if not visible:
			cut = sampcut.empty(d.ndet, d.nsamp)
		else:
			pmap = pmat.PmatMap(scan, mask, sys="sidelobe:%s" % args.objname)
			# Build a tod to project onto.
			tod = np.zeros((d.ndet, d.nsamp), dtype=dtype)
		# And project
			pmap.forward(tod, mask)
			# Any nonzero samples should be cut
			tod = tod != 0
			cut = sampcut.from_mask(tod)
			del tod
		progress = 100.0*(ind-comm.rank*ntod//comm.size)/((comm.rank+1)*ntod//comm.size-comm.rank*ntod//comm.size)
		print("%3d %5.1f %s %6.4f %d  %8.3f %8.3f" % (comm.rank, progress, id, float(cut.sum())/cut.size, visible, memory.current()/1024.**3, memory.max()/1024.**3))
		mystats.append([ind, float(cut.sum())/cut.size, visible])
		# Add to my work file
		_, uids  = actdata.split_detname(d.dets)
		flags = flagrange.from_sampcut(cut, dets=uids)
		flags.write(hfile, group=id)

# Merge all the individual cut files into a single big one.
comm.Barrier()
if comm.rank == 0:
	with h5py.File(args.odir + "/cuts.hdf", "w") as ofile:
		for i in range(comm.size):
			print("Reducing %3d" % i)
			with h5py.File(args.odir + "/work_%03d.hdf" % i, "r") as ifile:
				for key in sorted(ifile.keys()):
					ifile.copy(key, ofile)
	print("Done")

# Output the overall statistics
if len(mystats) == 0: mystats = [-1,0,0]
mystats = np.array(mystats)
stats = utils.allgatherv(mystats, comm)
if comm.rank == 0:
	with open(args.odir + "/stats.txt","w") as f:
		for stat in stats:
			if stat[0] >= 0:
				f.write("%s %6.4f %d\n" % (ids[int(stat[0])], stat[1], int(stat[2])))
