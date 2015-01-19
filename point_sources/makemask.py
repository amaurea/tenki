import numpy as np, argparse
from enlib import enmap
from scipy import ndimage
parser = argparse.ArgumentParser()
parser.add_argument("pos")
parser.add_argument("template")
parser.add_argument("ofile")
parser.add_argument("-r", "--radius", type=float, default=3)
parser.add_argument("-c", "--columns", type=str, default="3,5,2")
parser.add_argument("-t", "--threshold", type=float, default=0)
args = parser.parse_args()

cols = [int(w) for w in args.columns.split(",")]
srcinfo = np.loadtxt(args.pos)[:,cols]
pos = srcinfo[np.abs(srcinfo[:,2])>=args.threshold][:,:2] * np.pi/180
map = enmap.read_map(args.template)
pix = map.sky2pix(pos.T).T.astype(int)

pixrad = (map.area()/map.npix)**0.5
mrad = args.radius*np.pi/180/60/pixrad

mask = enmap.zeros(map.shape[-2:], map.wcs)+1
mask[pix[:,0],pix[:,1]] = 0
mask = enmap.enmap(1.0*(ndimage.distance_transform_edt(mask) > mrad), map.wcs)
enmap.write_map(args.ofile, mask)
