using ReTestItems

const BOLTZ_TEST_GROUP = lowercase(get(ENV, "BOLTZ_TEST_GROUP", "all"))
@info "Running tests for group: $(BOLTZ_TEST_GROUP)"

if BOLTZ_TEST_GROUP == "all"
    ReTestItems.runtests(@__DIR__)
else
    tag = Symbol(LUX_TEST_GROUP)
    ReTestItems.runtests(@__DIR__; tags=[tag])
end
