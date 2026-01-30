module TestUtils

using Reactant, Enzyme, Lux

sumabs2first(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

function compute_enzyme_gradient(model, x, ps, st)
    return compute_enzyme_gradient(sumabs2first, model, x, ps, st)
end

function compute_enzyme_gradient(f::F, model, x, ps, st) where {F}
    res = Enzyme.gradient(Reverse, f, Const(model), x, ps, Const(st))
    return res[2], res[3]
end

compute_reactant_gradient(args...) = @jit compute_enzyme_gradient(args...)

function compute_reactant_gradient_fd(model, x, ps, st)
    return compute_reactant_gradient_fd(sumabs2first, model, x, ps, st)
end

function compute_reactant_gradient_fd(f::F, model, x, ps, st) where {F}
    _, dx, dps, _ = @jit Reactant.TestUtils.finite_difference_gradient(
        f, Const(model), f64(x), f64(ps), Const(st)
    )
    return dx, dps
end

end
