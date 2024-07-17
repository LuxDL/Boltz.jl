using ReTestItems, Pkg

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

const BOLTZ_TEST_GROUP = lowercase(get(ENV, "BOLTZ_TEST_GROUP", "all"))
@info "Running tests for group: $BOLTZ_TEST_GROUP"
const RETESTITEMS_NWORKERS = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", "0"))

ReTestItems.runtests(
    @__DIR__; tags=(BOLTZ_TEST_GROUP == "all" ? nothing : [Symbol(BOLTZ_TEST_GROUP)]),
    nworkers=ifelse(BACKEND_GROUP ∈ ("cuda", "amdgpu"), 0, RETESTITEMS_NWORKERS))
