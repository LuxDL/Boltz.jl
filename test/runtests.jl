using ReTestItems, Pkg, InteractiveUtils, Hwloc

@info sprint(versioninfo)

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))
const EXTRA_PKGS = String[]

(BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") && push!(EXTRA_PKGS, "LuxCUDA")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") && push!(EXTRA_PKGS, "AMDGPU")

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS=EXTRA_PKGS
    Pkg.add(EXTRA_PKGS)
    Pkg.update()
    Base.retry_load_extensions()
    Pkg.instantiate()
end

using Boltz

const BOLTZ_TEST_GROUP = get(ENV, "BOLTZ_TEST_GROUP", "all")
const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 4))))
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))))

@info "Running tests for group: $BOLTZ_TEST_GROUP with $RETESTITEMS_NWORKERS workers"

ReTestItems.runtests(
    Boltz; tags=(BOLTZ_TEST_GROUP == "all" ? nothing : [Symbol(BOLTZ_TEST_GROUP)]),
    nworkers=ifelse(BACKEND_GROUP ∈ ("cuda", "amdgpu"), 0, RETESTITEMS_NWORKERS),
    nworker_threads=RETESTITEMS_NWORKER_THREADS, testitem_timeout=3600)
