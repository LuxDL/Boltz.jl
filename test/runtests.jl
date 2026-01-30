using Boltz, ParallelTestRunner

testsuite = find_tests(@__DIR__)
delete!(testsuite, "testutils")
delete!(testsuite, "vision/testutils")

# Limit total jobs to 4 to avoid OOM on GPU
total_jobs = min(min(ParallelTestRunner.default_njobs(), length(keys(testsuite))), 4)

testrunner_args = @isdefined(TEST_ARGS) ? TEST_ARGS : ARGS
push!(testrunner_args, "--jobs=$(total_jobs)")

withenv(
    "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (total_jobs + 0.1),
    "XLA_REACTANT_GPU_PREALLOCATE" => false,
) do
    runtests(Boltz, testrunner_args; testsuite)
end
