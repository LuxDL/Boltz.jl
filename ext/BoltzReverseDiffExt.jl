module BoltzReverseDiffExt

using ReverseDiff: ReverseDiff, TrackedArray, @grad_from_chainrules

using MLDataDevices: CPUDevice

using Boltz: Layers

@grad_from_chainrules Layers.apply_dynamic_expression(
    de::Layers.InternalDynamicExpressionWrapper,
    expr,
    operator_enum,
    x::TrackedArray,
    ps,
    ::CPUDevice,
)
@grad_from_chainrules Layers.apply_dynamic_expression(
    de::Layers.InternalDynamicExpressionWrapper,
    expr,
    operator_enum,
    x,
    ps::TrackedArray,
    ::CPUDevice,
)
@grad_from_chainrules Layers.apply_dynamic_expression(
    de::Layers.InternalDynamicExpressionWrapper,
    expr,
    operator_enum,
    x::TrackedArray,
    ps::TrackedArray,
    ::CPUDevice,
)

end
