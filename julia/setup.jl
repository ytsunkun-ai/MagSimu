# Install the Julia environment used by julia/hysteresis.jl.
#
# Run once:
#   julia --project=julia julia/setup.jl
#
# MicroMagnetic.jl currently requires Julia >= 1.11.  The Python pipeline never
# assumes that Julia is installed; it reports a fallback if this setup has not
# been performed.

using Pkg

Pkg.activate(@__DIR__)
Pkg.add(["MicroMagnetic", "JSON"])
Pkg.precompile()

println("Julia environment ready in ", @__DIR__)
