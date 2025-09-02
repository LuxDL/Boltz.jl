using ReTestItems, Pkg, Hwloc, Test

const ALL_BOTLZ_TEST_GROUPS = [
    "layers", "others", "vision", "vision_metalhead", "integration", "piml"
]

function parse_test_args()
    test_args_from_env = @isdefined(TEST_ARGS) ? TEST_ARGS : ARGS
    test_args = Dict{String,String}()
    for arg in test_args_from_env
        if contains(arg, "=")
            key, value = split(arg, "="; limit=2)
            test_args[key] = value
        end
    end
    @info "Parsed test args" test_args
    return test_args
end

const PARSED_TEST_ARGS = parse_test_args()

const BOLTZ_TEST_REACTANT = parse(
    Bool, lowercase(get(PARSED_TEST_ARGS, "BOLTZ_TEST_REACTANT", "true"))
)

INPUT_TEST_GROUP = lowercase(get(PARSED_TEST_ARGS, "BOLTZ_TEST_GROUP", "all"))
const BOLTZ_TEST_GROUP = if startswith("!", INPUT_TEST_GROUP[1])
    exclude_group = lowercase.(split(INPUT_TEST_GROUP[2:end], ","))
    filter(x -> x ∉ exclude_group, ALL_BOTLZ_TEST_GROUPS)
else
    [INPUT_TEST_GROUP]
end

const BACKEND_GROUP = lowercase(get(PARSED_TEST_ARGS, "BACKEND_GROUP", "all"))
const EXTRA_PKGS = String[]

(BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
    !BOLTZ_TEST_REACTANT &&
    push!(EXTRA_PKGS, "LuxCUDA")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
    !BOLTZ_TEST_REACTANT &&
    push!(EXTRA_PKGS, "AMDGPU")

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS = EXTRA_PKGS
    Pkg.add(EXTRA_PKGS)
    Pkg.update()
    Base.retry_load_extensions()
    Pkg.instantiate()
end

using Boltz

@testset "Boltz.jl Tests" begin
    @testset for (i, tag) in enumerate(BOLTZ_TEST_GROUP)
        withenv(
            "BOLTZ_TEST_REACTANT" => BOLTZ_TEST_REACTANT, "BACKEND_GROUP" => BACKEND_GROUP
        ) do
            ReTestItems.runtests(
                Boltz;
                tags=(tag == "all" ? nothing : [Symbol(tag)]),
                testitem_timeout=2400,
                nworkers=0,
                nworker_threads=parse(
                    Int,
                    get(
                        ENV,
                        "RETESTITEMS_NWORKER_THREADS",
                        string(Hwloc.num_virtual_cores()),
                    ),
                ),
            )
        end
    end
end
