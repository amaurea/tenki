import numpy as np, sys, os
from enlib import config, mpi, errors, log
from enact import filedb, actdata
config.default("verbosity", 1, "Verbosity of output")

# Fast and incremental mapping program.
# Idea:
#  1. Map in 2 steps: tod -> work and work -> tod
#  2. Work coordinates are pixel-shifted versions of sky coordinates,
#     with a different shift for each scanning pattern. Defined in
#     3 steps:
#      1. Shift in RA based on example sweep (fwd and back separately)
#      2. Subdivide pixels in dec to make them approximately equi-spaced
#         in azimuth.
#      3. Transpose for memory access efficiency.
#  3. Ignore detector correlations. That way the work-space noise matrix
#     is purely horizontal.
#  4. Use the same noise spectrum shape for all tods in a work-space. That
#     way the work-space will be characterized by a single pixel-space
#     noise matrix. This will not bias, just make us supoptimal. Can
#     still have different detector weights used on white noise level, though.
#     May have separate work-spaces for e.g. good and bad weather.
#  5. For each new tod, determine which work-space it belongs to, and
#     apply bw += Pt'Wt Ft Wt d and and Ww**2 += Pt'Wt**2 1, where Wt is the sqrt of the
#     sample weighting and Ft is the frequency filter, which is constant for this workspace,
#     and Pt is the tod-to-work pointing matrix. (Wt is just a number per detector, so
#     it commutes with Ft)
#  6. We model the work maps as mw = Pw m + n, cov(n)" = Ww Fw Ww
#     (Pw' Ww Fw Ww Pw)m = Pw' Ww Fw Ww mw approx Pw' bw
#     So we can solve the full map via CG on the work maps, all of which
#     are pixel-space operations, with no interpolation.
#  7. This map will be approximately unbiased if Pw' Ww Fw Ww mw is approximately Pw' bw.
#     Deviations come from:
#      1. Scans not mapping completely horizontally to work spaces
#      2. Detector are offset in az
#      3. Inexact mapping from t to x in work coordinates
#
# In theory we could have one work-space per scanning pattern, but
# that would make them very big. Instead, it makes sense to divide
# them into blocks by starting RA (RA of bottom-left corner of scan).

# How would this work in practice? Split operation into 3 steps:
# STEP 1: Classify
#  1. For each new tod-file, read in its pointing and determine its scanning
#     pattern. Possibly also read in the tod to get the noise correlation structure.
#  2. Determine a workspace-id, which is given by el-az1-az2-rablock-array-noise.
#     Rablock is an integer like int(sRA/15). Noise is an integer like int(log(spec(0.1Hz)/spec(10Hz)))
#  3. Output a file with lines of [tod] [workspace-id]
# STEP 2: Build
#  1. Read in file from previous step, and group tods by wid.
#  2. For each wid, create its metadata. This should be fully specified by the
#     workspace-id + settings, so multiple runs create compatible workspaces. So
#     reading in a TOD should not be necessary for this.
#     Metadata is: workspace area, pixel shifts, frequency filter and t-x scaling.
#  3. For each tod in group, read and calibrate it. Then measure the noise spec
#     per det. Get white noise level for det-weighting, and check how well noise
#     profile matches freq filter. Cut outliers (tradeoff).
#  4. Project into our workspace rhs and div, adding it to the existing ones.
# STEP 3: Solve
#  1. Loop through workspaces and build up bm = Pw' bw and diag(Wm) = Pw' diag(Ww)
#  2. Solve system through CG-iteration.
#
# Should we maintain a single set of workspaces that we keep updating with more data?
# Or should we create a new set of workspaces per week, say? The latter will take more
# disk space, but makes it possible to produce week-wise maps and remove any glitch
# weeks. As long as the workspace-ids are fully deterministic, weeks can still be easily
# coadded later.
#
# I prefer the latter. It makes each run independent, and you don't risk losing data
# by making an error while updating the existing files.

if len(sys.argv) < 2:
	sys.stderr.write("Usage python fastmap.py [command], where command is classify, build or solve\n")
	sys.exit(1)

command = sys.argv[1]
comm    = mpi.COMM_WORLD

if command == "classify":
	parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
	parser.add_argument("command")
	parser.add_argument("sel")
	args = parser.parse_args()

	log_level = log.verbosity2level(config.get("verbosity"))
	L = log.init(level=log_level, rank=comm.rank)

	filedb.init()
	ids = filedb.scans[args.sel]
	for ind in range(comm.rank, len(ids), comm.size):
		id    = ids[ind]
		entry = filedb.data[id]
		try:
			# We need the tod and all its dependences to estimate which noise
			# category the tod falls into. But we don't need all the dets.
			# Speed things up by only reading 25% of them.
			d = actdata.read(entry, ["boresight","tags","point_offsets","gain","polangle","site","cut_noiseest","layout","tod_shape"])
			d.restrict(dets=d.dets[::4])
			d += actdata.read_tod(entry, dets=d.dets)
			d = actdata.calibrate(d, exclude=["autocut"])
			if d.ndet == 0 or d.nsamp == 0:
				raise errors.DataMissing("Tod contains no valid data")
		except errors.DataMissing as e:
			if not quiet: L.debug("Skipped %s (%s)" % (str(filelist[ind]), e.message))
			continue
		L.debug(id)



