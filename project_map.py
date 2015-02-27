import argparse
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("template")
parser.add_argument("omap")
parser.add_argument("--order", type=int, default=3)
parser.add_argument("--mode", type=str, default="constant")
args = parser.parse_args()

m = enmap.read_map(args.imap)
t = enmap.read_map(args.template)
o = enmap.project(m, t.shape, t.wcs, order=args.order, mode=args.mode)
enmap.write_map(args.omap, o)
