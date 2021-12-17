import numpy as np, os, h5py
from enlib import config, resample, utils, gapfill
from enact import actdata, filedb

parser = config.ArgumentParser()
parser.add_argument("query")
parser.add_argument("ofile")
parser.add_argument("-d", "--dets", type=str, default=None)
parser.add_argument("-D", "--absdets", type=str, default=None)
parser.add_argument("-c", "--calib", action="store_true")
parser.add_argument("-C", "--manual-calib", type=str, default=None)
parser.add_argument("--bin", type=int, default=1)
parser.add_argument("--nofft", action="store_true")
parser.add_argument("--nodeslope", action="store_true")
parser.add_argument("-F", "--fields", type=str, default=None)
args = parser.parse_args()

filedb.init()
ids = filedb.scans[args.query]
if len(ids) > 1:
	# Will process multiple files
	utils.mkdir(args.ofile)
for id in ids:
	print(id)
	entry = filedb.data[id]
	subdets = None
	absdets = None
	if args.absdets is not None:
		absdets = [int(w) for w in args.absdets.split(",")]
	elif args.dets is not None:
		subdets = [int(w) for w in args.dets.split(",")]

	fields = ["gain","tconst","cut","tod","boresight"]
	if args.fields: fields = args.fields.split(",")
	d = actdata.read(entry, fields=fields)
	if absdets: d.restrict(dets=absdets)
	if subdets: d.restrict(dets=d.dets[subdets])
	if args.calib: d = actdata.calibrate(d, exclude=["autocut"])
	elif args.manual_calib:
		ops = args.manual_calib.split(",")
		if "safe" in ops: d.boresight[1:] = utils.unwind(d.boresight[1:], period=360)
		if "rad" in ops: d.boresight[1:] *= np.pi/180
		if "bgap" in ops:
			bad = (d.flags!=0)*(d.flags!=0x10)
			for b in d.boresight: gapfill.gapfill_linear(b, bad, inplace=True)
		if "gain" in ops: d.tod *= d.gain[:,None]
		if "tgap" in ops: 
			gapfiller = {"copy":gapfill.gapfill_copy, "linear":gapfill.gapfill_linear}[config.get("gapfill")]
			gapfiller(d.tod, d.cut, inplace=True)
		if "slope" in ops:
			utils.deslope(d.tod, w=8, inplace=True)
		if "deconv" in ops:
			d = actdata.calibrate(d, operations=["tod_fourier"])
	if args.bin > 1:
		d.tod = resample.downsample_bin(d.tod, steps=[args.bin])
		if "boresight" in d:
			d.boresight = resample.downsample_bin(d.boresight, steps=[args.bin])
		if "flags" in d:
			d.flags = resample.downsample_bin(d.flags, steps=[args.bin])
	oname = args.ofile
	if len(ids) > 1: oname = "%s/%s.hdf" % (args.ofile, id)
	with h5py.File(oname, "w") as hfile:
		if "tod" in d: hfile["tod"] = d.tod
		if "boresight" in d:
			hfile["az"]  = d.boresight[1]
			hfile["el"]  = d.boresight[2]
		hfile["dets"] = np.char.encode(d.dets)
		try:
			hfile["mask"] = d.cut.to_mask().astype(np.int16)
		except AttributeError: pass
