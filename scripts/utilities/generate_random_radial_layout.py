import math

import matplotlib.pyplot as plt
import numpy

# Define minimum station separation.
min_sep = 35 # metres

# Core area.
# num_points = 224+72
# max_radius = 1700 # metres

# Whole telescope.
num_points = 512
max_radius = 40000 # metres

# Generate normalised radial profile according to chosen distribution.
r = numpy.random.lognormal(sigma=1.0, size=num_points)

# Scale profile to maximum radius.
r = numpy.sort(r)
r *= (max_radius / numpy.max(r))

# Allocate space for modified x,y coordinates. 
x = numpy.zeros(num_points)
y = numpy.zeros(num_points)

# Loop over radial positions to generate non-overlapping x,y coordinates.
for i in range(num_points):
    print("Checking station %d" % (i))
    trial = 0
    while True:
        # Generate theta (uniform from 0 to 2pi)
        theta = 2.0 * math.pi * numpy.random.uniform()
        t_x = r[i] * math.cos(theta)
        t_y = r[i] * math.sin(theta)

        # Check distance to all other stations up to this one.
        min_dist = 1e100
        for j in range(i):
            d_x = x[j] - t_x
            d_y = y[j] - t_y
            d = math.sqrt(d_x*d_x + d_y*d_y)
            if d < min_dist:
                min_dist = d

        # If minimum distance is greater than the required minimum separation, 
        # store coordinates and go to the next point.
        # Otherwise, keep trying.
        if min_dist >= min_sep:
            x[i] = t_x
            y[i] = t_y
            break
        else:
            trial += 1

        # Check if we've exceeded the maximum number of trials at this radius.
        if trial > 500:
            r[i] += 1
            print("  Increasing radius to %.2f m" % (r[i]))

# Plot
plt.scatter(x,y)
plt.show()
