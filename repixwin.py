import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifile")
parser.add_argument("ofile")
parser.add_argument("--iwin", type=int, default=1)
parser.add_argument("--owin", type=int, default=0)
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils

def repixwin(imap, iwin, owin, rapod=5*utils.arcmin):
	mask  = imap.preflat[0]!=0
	apod  = enmap.apod_mask(mask, rapod)
	imap *= apod
	fmap  = enmap.fft(imap)
	del imap
	iwy, iwx = enmap.calc_window(fmap.shape, order=iwin)
	owy, owx = enmap.calc_window(fmap.shape, order=owin)
	fmap *= (owy/iwy)[:,None]
	fmap *= (owx/iwx)
	omap = enmap.ifft(fmap).real
	del fmap
	omap *= mask
	# Undo apodization
	mask  = apod>0
	omap[...,mask] /= apod[mask]
	return omap
map = enmap.read_map(args.ifile)
map = repixwin(map, args.iwin, args.owin)
enmap.write_map(args.ofile, map)
