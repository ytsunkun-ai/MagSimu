# Build a small cluster shape for the representative sub-volume job.
#
# This is a geometric placeholder for a particle cluster, not a DEM-derived
# agglomerate.  The Python side explicitly reports that limitation.

function make_cluster(n::Int, radius::Float64, spacing::Float64)
    n >= 1 || error("n must be >= 1")
    shape = Sphere(center=(0.0, 0.0, 0.0), radius=radius)
    # A compact, deterministic arrangement: centre plus two rings.  It keeps the
    # cluster inside the mesh while exercising inter-particle exchange/demag.
    ring1 = 6
    for k in 1:min(n - 1, ring1)
        theta = 2pi * (k - 1) / ring1
        center = (spacing * cos(theta), spacing * sin(theta), 0.0)
        shape = shape + Sphere(center=center, radius=radius)
    end
    for k in 1:(n - 1 - min(n - 1, ring1))
        theta = 2pi * (k - 1) / max(n - 1 - ring1, 1)
        center = (1.8 * spacing * cos(theta), 1.8 * spacing * sin(theta), 0.0)
        shape = shape + Sphere(center=center, radius=radius)
    end
    return shape
end
