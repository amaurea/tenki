import numpy as np, argparse
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("div")
parser.add_argument("-t", "--thin", type=int, default=1000)
args = parser.parse_args()

div = enmap.read_map(args.div)
div = div.reshape((-1,)+div.shape[-2:])[0]
pix_area = div.area()*(180*60/np.pi)**2

rms  = (pix_area/div)**0.5
rms  = np.array(np.sort(rms, axis=None))
area = np.arange(rms.size)*pix_area/3600

np.savetxt("/dev/stdout/", np.array([area[::args.thin],rms[::args.thin]]).T, fmt="%9.3f %15.7e")
