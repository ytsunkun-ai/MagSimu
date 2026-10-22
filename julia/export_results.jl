# Convert a raw loop CSV into the same small JSON metadata contract used by the
# main job.  This utility is useful when a MicroMagnetic run is performed
# interactively and only a CSV is retained.
#
# Usage:
#   julia --project=julia julia/export_results.jl --input loop.csv --output result.json --ms 1.25e6

using DelimitedFiles
using JSON

function parse_args(args)
    input = ""
    output = "result.json"
    ms = 0.0
    i = 1
    while i <= length(args)
        if args[i] == "--input" && i < length(args)
            input = args[i + 1]; i += 2
        elseif args[i] == "--output" && i < length(args)
            output = args[i + 1]; i += 2
        elseif args[i] == "--ms" && i < length(args)
            ms = parse(Float64, args[i + 1]); i += 2
        else
            i += 1
        end
    end
    isempty(input) && error("missing --input")
    return input, output, ms
end

input, output, ms = parse_args(ARGS)
raw, _ = readdlm(input, ',', header=true)
data = Float64.(raw)
meta = Dict(
    "schema_version" => "1.0",
    "model_status" => "conditional",
    "engine" => "MicroMagnetic.jl",
    "source_csv" => input,
    "n_points" => size(data, 1),
    "Ms_A_m" => ms,
)
open(output, "w") do io
    JSON.print(io, meta, 2)
    println(io)
end
println("wrote ", output)
