import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.visualization import simple_norm

from photutils.centroids import centroid_quadratic
from photutils.datasets import make_noise_image
from photutils.profiles import RadialProfile

# create an artificial single source
gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
yy, xx = np.mgrid[0:100, 0:100]
data = gmodel(xx, yy)
error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
data += error

# find the source centroid
xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

# create the radial profile
edge_radii = np.arange(26)
rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

norm = simple_norm(data, 'sqrt')
plt.figure(figsize=(5, 5))
plt.imshow(data, norm=norm)
rp.apertures[5].plot(color='C0', lw=2)
rp.apertures[10].plot(color='C1', lw=2)