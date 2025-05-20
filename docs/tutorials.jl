const ALL_TUTORIALS = [
    ("GettingStarted/main.jl", true),
    ("SymbolicOptimalControl/main.jl", false)
]

const TUTORIALS = collect(enumerate(ALL_TUTORIALS))

const BUILDKITE_PARALLEL_JOB_COUNT = parse(
    Int, get(ENV, "BUILDKITE_PARALLEL_JOB_COUNT", "-1")
)

const TUTORIALS_BUILDING = if BUILDKITE_PARALLEL_JOB_COUNT > 0
    id = parse(Int, ENV["BUILDKITE_PARALLEL_JOB"]) + 1 # Index starts from 0
    splits = collect(
        Iterators.partition(
            TUTORIALS, cld(length(TUTORIALS), BUILDKITE_PARALLEL_JOB_COUNT)
        ),
    )
    id > length(splits) ? [] : splits[id]
else
    TUTORIALS
end

const NTASKS = min(
    parse(Int, get(ENV, "LUXLIB_DOCUMENTATION_NTASKS", "1")), length(TUTORIALS_BUILDING)
)

@info "Building Tutorials:" TUTORIALS_BUILDING

@info "Starting Lux Tutorial Build with $(NTASKS) tasks."

run(
    `$(Base.julia_cmd()) --startup=no --code-coverage=user --threads=$(Threads.nthreads()) --project=@literate -e 'import Pkg; Pkg.add(["Literate", "InteractiveUtils"])'`,
)

asyncmap(TUTORIALS_BUILDING; ntasks=NTASKS) do (i, (p, should_run))
    @info "Running Tutorial $(i): $(p) on task $(current_task())"
    path = joinpath(@__DIR__, "..", "examples", p)
    name = "$(i)_$(first(rsplit(p, "/")))"
    output_directory = joinpath(@__DIR__, "src", "tutorials")
    tutorial_proj = dirname(path)
    file = joinpath(dirname(@__FILE__), "run_single_tutorial.jl")

    withenv(
        "JULIA_NUM_THREADS" => "$(Threads.nthreads())",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(100 ÷ NTASKS)%",
        "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        "JULIA_DEBUG" => "Literate",
    ) do
        run(`$(Base.julia_cmd()) --color=yes --code-coverage=user --threads=$(Threads.nthreads()) --project=$(tutorial_proj) "$(file)" "$(name)" "$(output_directory)" "$(path)" "$(should_run)"`)
    end

    return nothing
end
