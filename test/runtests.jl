using ReTestItems, Pkg, Hwloc, Test

const ALL_BOTLZ_TEST_GROUPS = [
    "layers", "others", "vision", "vision_metalhead", "integration", "piml"
]

const BOLTZ_TEST_REACTANT = parse(Bool, lowercase(get(ENV, "BOLTZ_TEST_REACTANT", "true")))

INPUT_TEST_GROUP = lowercase(get(ENV, "BOLTZ_TEST_GROUP", "all"))
const BOLTZ_TEST_GROUP = if startswith("!", INPUT_TEST_GROUP[1])
    exclude_group = lowercase.(split(INPUT_TEST_GROUP[2:end], ","))
    filter(x -> x ∉ exclude_group, ALL_BOTLZ_TEST_GROUPS)
else
    [INPUT_TEST_GROUP]
end

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))
const EXTRA_PKGS = String[]

if "all" ∈ BOLTZ_TEST_GROUP || "integration" ∈ BOLTZ_TEST_GROUP
    # TODO: enable once https://github.com/SciML/DataInterpolations.jl/pull/414  lands
    # append!(EXTRA_PKGS, ["DataInterpolations"])
    # TODO: enable once https://github.com/SymbolicML/DynamicExpressions.jl/pull/119 lands
    append!(EXTRA_PKGS, ["DynamicExpressions"])
end
if "all" ∈ BOLTZ_TEST_GROUP || "vision_metalhead" ∈ BOLTZ_TEST_GROUP
    push!(EXTRA_PKGS, "Metalhead")
end

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

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 4)))
)

@testset "Boltz.jl Tests" begin
    @testset "[$(tag)] [$(i)/$(length(BOLTZ_TEST_GROUP))]" for (i, tag) in
                                                               enumerate(BOLTZ_TEST_GROUP)
        nworkers = ifelse(
            BACKEND_GROUP ∈ ("cuda", "amdgpu") &&
            (tag == "vision" || tag == "vision_metalhead"),
            0,
            RETESTITEMS_NWORKERS,
        )
        nworker_threads = parse(
            Int,
            get(
                ENV,
                "RETESTITEMS_NWORKER_THREADS",
                string(max(Hwloc.num_virtual_cores() ÷ max(nworkers, 1), 1)),
            ),
        )

        ReTestItems.runtests(
            Boltz;
            tags=(tag == "all" ? nothing : [Symbol(tag)]),
            testitem_timeout=2400,
            nworkers,
            nworker_threads,
        )
    end
end
