import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ofile")
parser.add_argument("-r", "--ref",   type=str, default=None)
parser.add_argument("-m", "--maps",  type=str, default=None)
parser.add_argument("-v", "--ivars", type=str, default=None)
parser.add_argument("-b", "--beam",  type=str, default=None)
parser.add_argument("-i", "--info",  type=str, default=None)
parser.add_argument("-g", "--gain",  type=float, default=None)
parser.add_argument("-f", "--freq",  type=str, default=None, help="Frequency (GHz) or act band tag")
parser.add_argument("-c", "--copy",  action="store_true")
args = parser.parse_args()
from enlib import mapdata

mode  = "copy" if args.copy else "link"
maps  = args.maps .split(",") if args.maps  else None
ivars = args.ivars.split(",") if args.ivars else None

mapinfo = mapdata.build_mapinfo(mapfiles=maps, ivarfiles=ivars,
		beamfile=args.beam, infofile=args.info, gain=args.gain, freq=args.freq,
		mapdatafile=args.ref)

if   mapinfo.maps  is None: print("Maps missing!")
elif mapinfo.ivars is None: print("Ivars missing!")
elif mapinfo.beam  is None: print("Beam missing")
elif mapinfo.freq  is None: print("Freq missing")

# Handy shortcuts
freqs = {"f030": 27, "f040": 39, "f090": 98, "f150": 150, "f220": 220}
try: mapinfo.freq = float(freqs[mapinfo.freq])
except KeyError: mapinfo.freq = float(mapinfo.freq)

mapdata.write_mapinfo(args.ofile, mapinfo, mode=mode)
