# MicroMagnetic.jl main hysteresis job.
#
# Usage:
#   julia --project=julia julia/hysteresis.jl --input job.json --output outdir
#
# The Python driver creates job.json; this script is deliberately self-contained
# and writes loop.csv plus result.json.  Units at the Julia/Python boundary are SI
# (A/m, A/m, T).  MicroMagnetic's `hysteresis` returns H in mT and normalized m;
# this script converts both explicitly.

using JSON
using MicroMagnetic
using DelimitedFiles
using Printf

const MU0 = 4.0e-7 * pi

function parse_args(args)
    input = ""
    output = ""
    i = 1
    while i <= length(args)
        if args[i] == "--input" && i < length(args)
            input = args[i + 1]
            i += 2
        elseif args[i] == "--output" && i < length(args)
            output = args[i + 1]
            i += 2
        else
            i += 1
        end
    end
    isempty(input) && error("missing --input job.json")
    isempty(output) && error("missing --output directory")
    return input, output
end

function write_loop_csv(path, H_A_m, M_A_m)
    open(path, "w") do io
        println(io, "H_A_m,M_phase_A_m")
        for i in eachindex(H_A_m)
            @printf(io, "%.12e,%.12e\n", H_A_m[i], M_A_m[i])
        end
    end
end

function main()
    input, output = parse_args(ARGS)
    mkpath(output)
    job = JSON.parsefile(input)
    material = job["material"]
    mesh_cfg = job["mesh"]
    sweep = job["sweep"]
    sim_cfg = job["simulation"]
    shape_cfg = job["shape"]

    Ms = Float64(material["Ms_A_m"])
    A = Float64(material["exchange_A_J_m"])
    K = Float64(material["anisotropy_K_J_m3"])
    alpha = Float64(material["damping_alpha"])

    mesh = FDMesh(;
        nx=Int(mesh_cfg["nx"]), ny=Int(mesh_cfg["ny"]), nz=Int(mesh_cfg["nz"]),
        dx=Float64(mesh_cfg["dx_m"]), dy=Float64(mesh_cfg["dy_m"]),
        dz=Float64(mesh_cfg["dz_m"]),
    )
    extent = (mesh_cfg["nx"] * mesh_cfg["dx_m"],
              mesh_cfg["ny"] * mesh_cfg["dy_m"],
              mesh_cfg["nz"] * mesh_cfg["dz_m"])
    sim = Sim(mesh; driver=sim_cfg["driver"], integrator=sim_cfg["integrator"],
              name="softmag", save_data=false)
    set_alpha(sim, alpha)

    warnings = String[]
    shape_note = "uniform box sub-volume"
    if shape_cfg["model"] == "sphere"
        d50 = shape_cfg["d50_m"]
        r = d50 === nothing ? 0.35 * minimum(extent) :
             min(0.5 * Float64(d50), 0.45 * minimum(extent))
        set_Ms(sim, Sphere(center=(0.0, 0.0, 0.0), radius=r), Ms)
        shape_note = @sprintf("sphere radius %.3e m (clipped to mesh sub-volume)", r)
        push!(warnings, "sphere 仅覆盖代表性子体积，不能代表完整颗粒。")
    elseif shape_cfg["model"] == "cluster"
        include(joinpath(@__DIR__, "particle_cluster.jl"))
        r = 0.14 * minimum(extent)
        shape = make_cluster(Int(shape_cfg["n_cluster_particles"]), r, 2.3 * r)
        set_Ms(sim, shape, Ms)
        shape_note = @sprintf("cluster of %d spheres, radius %.3e m",
                              Int(shape_cfg["n_cluster_particles"]), r)
        push!(warnings, "cluster 是网格可容纳的示意性团聚体，不是完整 DEM 团簇。")
    else
        set_Ms(sim, Ms)
        push!(warnings, "box 是颗粒内部代表性子体积，不是完整颗粒。")
    end

    A > 0 && add_exch(sim, A)
    sim_cfg["demag"] && add_demag(sim)
    K > 0 && add_anis(sim, K; axis=(1.0, 0.0, 0.0))
    add_zeeman(sim, (0.0, 0.0, 0.0))
    init_m0(sim, Tuple(Float64.(sim_cfg["m0"])))

    Hs = Float64.(sweep["H_A_m"])
    full_loop = Bool(sweep["full_loop"])
    # MicroMagnetic's hysteresis returns Nothing when full_loop=false.  We always
    # compute a full loop and truncate the returned data for up_only jobs.
    H_mT, m = hysteresis(
        sim, Hs;
        direction=Tuple(Float64.(sweep["direction"])),
        stopping_dmdt=Float64(sim_cfg["stopping_dmdt"]),
        max_steps=10000,
        output="none",
        full_loop=true,
    )
    n_forward = length(Hs)
    if !full_loop
        H_mT = H_mT[1:n_forward]
        m = m[1:n_forward]
    end
    H_A_m = Float64.(H_mT) .* (1.0e-3 / MU0)
    M_A_m = Float64.(m) .* Ms

    write_loop_csv(joinpath(output, "loop.csv"), H_A_m, M_A_m)
    meta = Dict(
        "schema_version" => "1.0",
        "model_status" => "conditional",
        "engine" => "MicroMagnetic.jl",
        "package_version" => string(pkgversion(MicroMagnetic)),
        "julia_version" => string(VERSION),
        "mesh" => mesh_cfg,
        "shape" => Dict("model" => shape_cfg["model"], "description" => shape_note),
        "n_points" => length(H_A_m),
        "Ms_A_m" => Ms,
        "exchange_A_J_m" => A,
        "anisotropy_K_J_m3" => K,
        "damping_alpha" => alpha,
        "sweep" => sweep,
        "warnings" => warnings,
    )
    open(joinpath(output, "result.json"), "w") do io
        JSON.print(io, meta, 2)
        println(io)
    end
    @info "MicroMagnetic hysteresis finished" n_points=length(H_A_m) output=output
end

try
    main()
catch err
    # Write a machine-readable error even when the package/API fails, then
    # propagate the non-zero exit code so Python can decide to fall back.
    input, output = try
        parse_args(ARGS)
    catch
        ("", "")
    end
    if !isempty(output)
        mkpath(output)
        open(joinpath(output, "result.json"), "w") do io
            JSON.print(io, Dict(
                "schema_version" => "1.0",
                "model_status" => "failed",
                "engine" => "MicroMagnetic.jl",
                "error" => sprint(showerror, err),
            ), 2)
            println(io)
        end
    end
    rethrow(err)
end
