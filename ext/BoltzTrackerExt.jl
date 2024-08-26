module BoltzTrackerExt

using Tracker: Tracker, TrackedArray, TrackedReal, @grad_from_chainrules

using MLDataDevices: CPUDevice

using Boltz: Layers

for T1 in (:TrackedArray, :AbstractArray), T2 in (:TrackedArray, :AbstractArray)
    T1 === :AbstractArray && T2 === :AbstractArray && continue

    @eval @grad_from_chainrules Layers.apply_dynamic_expression(
        de::Layers.InternalDynamicExpressionWrapper, expr, operator_enum, x::$(T1),
        ps::$(T2), dev::CPUDevice)
end

end
