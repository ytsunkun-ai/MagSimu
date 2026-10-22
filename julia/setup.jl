# Install / verify the Julia environment used by julia/hysteresis.jl.
#
# Run once from the repository root:
#   julia --project=julia julia/setup.jl
#
# The environment is pinned by julia/Project.toml and julia/Manifest.toml.
# MicroMagnetic.jl 0.6.x currently requires Julia >= 1.11.  The Python pipeline
# never assumes that Julia is installed; it reports a documented fallback if
# this setup has not been performed.

using Pkg

project_dir = @__DIR__
Pkg.activate(project_dir)

# Prevent Pkg.instantiate from launching a precompile workload in the caller's
# working directory.  The workload is run explicitly below in a temporary
# directory so generated OVF/TXT side products cannot pollute the repository.
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
Pkg.instantiate()

cd(mktempdir()) do
    Pkg.precompile()
end

using MicroMagnetic
using JSON

println("Julia environment ready in ", project_dir)
println("Julia:         ", VERSION)
println("MicroMagnetic: ", pkgversion(MicroMagnetic))
println("JSON:          ", pkgversion(JSON))
Pkg.status()
