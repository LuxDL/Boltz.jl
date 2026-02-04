using Boltz, ParallelTestRunner, Setfield

parsed_args = parse_args(@isdefined(TEST_ARGS) ? TEST_ARGS : ARGS)

testsuite = find_tests(@__DIR__)
delete!(testsuite, "testutils")
delete!(testsuite, "vision/testutils")

# Limit total jobs to 4 to avoid OOM on GPU
total_jobs = min(
    something(parsed_args.jobs, ParallelTestRunner.default_njobs()),
    length(keys(testsuite)),
    4,
)

@set! parsed_args.jobs = Some(total_jobs)

withenv(
    "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (total_jobs + 0.1),
    "XLA_REACTANT_GPU_PREALLOCATE" => false,
    "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(100 / (total_jobs + 0.1))%",
) do
    runtests(Boltz, parsed_args; testsuite)
end
