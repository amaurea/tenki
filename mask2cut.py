import numpy as np, os, bunch
from enlib import utils, config, rangelist, mpi, pmat, scan as enscan, enmap, coordinates, errors
from enact import filedb, actdata, files
parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("sel")
parser.add_argument("objname")
parser.add_argument("mask")
parser.add_argument("odir")
parser.add_argument("-M", "--margin", type=float, default=0)
parser.add_argument("-s", "--persample", action="store_true")
args = parser.parse_args()

filedb.init()
ids  = filedb.scans[args.sel]
comm = mpi.COMM_WORLD
dtype= np.float32
margin = args.margin*utils.degree

imask = enmap.read_map(args.mask)
# Expand to 3 components, as the pointing code expects that
mask = enmap.zeros((3,)+imask.shape[-2:], imask.wcs, dtype)
mask[0] = imask.reshape((-1,)+imask.shape[-2:])[0]
del imask

utils.mkdir(args.odir)

myinds = np.arange(comm.rank, len(ids), comm.size)
mystats = []
for ind in myinds:
	id = ids[ind]
	entry = filedb.data[id]
	# We only need pointing to build this cut
	try:
		d = actdata.read(entry, ["point_offsets","boresight","site","array_info"])
		d = actdata.calibrate(d, exclude=["autocut"])
	except (errors.DataMissing, AttributeError) as e:
		print "Skipping %s (%s)" % (id, e.message)
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
	object_pos = coordinates.interpol_pos("cel","hor", args.objname, mjd, site=scan.site)
	visible = np.any(object_pos[1] >= margin)
	if not visible:
		cut = rangelist.zeros((d.ndet,d.nsamp))
	else:
		pmap = pmat.PmatMap(scan, mask, sys="hor:%s" % args.objname)
		# Build a tod to project onto.
		tod = np.zeros((d.ndet, d.nsamp), dtype=dtype)
		# And project
		pmap.forward(tod, mask)
		# Any nonzero samples should be cut
		tod = np.rint(tod)
		cut = rangelist.Multirange([rangelist.Rangelist(t) for t in tod])
	print "%s %6.4f %d" % (id, float(cut.sum())/cut.size, visible)
	mystats.append([ind, float(cut.sum())/cut.size, visible])
	# Write cuts to output directory
	if args.persample:
		files.write_cut("%s/%s.cuts" % (args.odir, id), d.dets, cut, nrow=d.array_info.nrow, ncol=d.array_info.ncol)
mystats = np.array(mystats)
stats = utils.allgatherv(mystats, comm)
if comm.rank == 0:
	with open(args.odir + "/stats.txt","w") as f:
		for stat in stats:
			f.write("%s %6.4f %d\n" % (ids[int(stat[0])], stat[1], int(stat[2])))
