@concrete struct MBConv <: AbstractLuxWrapperLayer{:layer}
    layer
end

function MBConv(
    in_channels, out_channels, kernel, stride; expansion_ratio, se_ratio, skip_connection
)
    @assert 0 < se_ratio ≤ 1

    mid_channels = ceil(Int, in_channels * expansion_ratio)
    expansion = if !isone(expansion_ratio)
        Chain(
            Conv((1, 1), in_channels => mid_channels; use_bias=false, pad=SamePad()),
            BatchNorm(mid_channels, swish),
        )
    else
        NoOpLayer()
    end

    depthwise = Chain(
        Conv(
            kernel,
            mid_channels => mid_channels;
            use_bias=false,
            stride,
            pad=SamePad(),
            groups=mid_channels,
        ),
        BatchNorm(mid_channels, swish),
    )

    n_squeezed_channels = max(1, ceil(Int, in_channels * se_ratio))
    excitation = Chain(
        AdaptiveMeanPool((1, 1)),
        Conv((1, 1), mid_channels => n_squeezed_channels, swish; pad=SamePad()),
        Conv((1, 1), n_squeezed_channels => mid_channels, σ; pad=SamePad()),
    )
    projection = Chain(
        Conv((1, 1), mid_channels => out_channels; pad=SamePad(), use_bias=false),
        BatchNorm(out_channels),
    )

    do_skip = skip_connection && stride == 1 && in_channels == out_channels

    return MBConv(
        SkipConnection(
            Chain(;
                expansion,
                depthwise,
                excitation=SkipConnection(excitation, Base.BroadcastFunction(*)),
                projection,
            ),
            do_skip ? Base.BroadcastFunction(+) : ((mx, x) -> mx),
        ),
    )
end

"""
    EfficientNet(variant::Union{String,Symbol}; pretrained::Bool=false, in_channels::Int=3,
                 nclasses::Int=1000)

Create an EfficientNet model [tan2019efficientnet](@citep).

## Arguments

  - `variant::Union{String,Symbol}`: The variant of the EfficientNet model to use. Valid
    variants are `:b0`, `:b1`, `:b2`, `:b3`, `:b4`, `:b5`, `:b6`, `:b7`.

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
  - `in_channels::Int=3`: The number of input channels. (Must be `3` for pretrained models.)
  - `nclasses::Int=1000`: The number of output classes. (Must be `1000` for pretrained
    models.)
"""
@concrete struct EfficientNet <: AbstractLuxVisionLayer
    layer
    pretrained_name::Symbol
    pretrained::Bool
    pretrained_format::Symbol
end

function EfficientNet(
    block_params;
    width_coefficient,
    depth_coefficient,
    nclasses::Int,
    depth_divisor::Int,
    min_depth::Union{Nothing,Int},
    include_top::Bool,
    include_head::Bool=true,
    in_channels=3,
    pretrained::Bool=false,
    pretrained_name=:unknown,
)
    out_channels = efficient_net_round_filter(
        32, width_coefficient, depth_divisor, min_depth
    )
    stem = Chain(
        Conv((3, 3), in_channels => out_channels; use_bias=false, stride=2, pad=SamePad()),
        BatchNorm(out_channels, swish),
    )

    blocks = MBConv[]
    for bp in block_params
        in_channels = efficient_net_round_filter(
            bp.in_channels, width_coefficient, depth_divisor, min_depth
        )
        out_channels = efficient_net_round_filter(
            bp.out_channels, width_coefficient, depth_divisor, min_depth
        )
        repeat =
            depth_coefficient ≈ 1 ? bp.repeat : ceil(Int64, depth_coefficient * bp.repeat)

        push!(
            blocks,
            MBConv(
                in_channels,
                out_channels,
                bp.kernel,
                bp.stride;
                bp.expansion_ratio,
                bp.se_ratio,
                bp.skip_connection,
            ),
        )
        for _ in 1:(repeat - 1)
            push!(
                blocks,
                MBConv(
                    out_channels,
                    out_channels,
                    bp.kernel,
                    1;
                    bp.expansion_ratio,
                    bp.se_ratio,
                    bp.skip_connection,
                ),
            )
        end
    end
    blocks = Chain(blocks...)

    if !include_head
        return EfficientNet(Chain(; stem, blocks), pretained_name, pretrained, :pth)
    end

    head_out_channels = efficient_net_round_filter(
        1280, width_coefficient, depth_divisor, min_depth
    )

    head = Chain(
        Conv((1, 1), out_channels => head_out_channels; use_bias=true, pad=SamePad()),
        BatchNorm(head_out_channels, swish),
    )

    top = include_top ? Dense(head_out_channels, nclasses) : NoOpLayer()

    return EfficientNet(
        Chain(;
            stem, blocks, head, pool=AdaptiveMeanPool((1, 1)), flatten=FlattenLayer(), top
        ),
        pretrained_name,
        pretrained,
        :pth,
    )
end

function EfficientNet(variant::Union{String,Symbol}; pretrained::Bool=false, kwargs...)
    model_name = Symbol(:efficientnet_, variant)
    @assert model_name in keys(EFFICIENTNET_CONFIG) "Unknown model name: $model_name"
    block_params, global_params = get_efficient_net_config(model_name; kwargs...)
    return EfficientNet(
        block_params; global_params..., pretrained, pretrained_name=model_name, kwargs...
    )
end

# width_coefficient, depth_coefficient, resolution
const EFFICIENTNET_CONFIG = Dict(
    :efficientnet_b0 => (1.0, 1.0, 224),
    :efficientnet_b1 => (1.0, 1.1, 240),
    :efficientnet_b2 => (1.1, 1.2, 260),
    :efficientnet_b3 => (1.2, 1.4, 300),
    :efficientnet_b4 => (1.4, 1.8, 380),
    :efficientnet_b5 => (1.6, 2.2, 456),
    :efficientnet_b6 => (1.8, 2.6, 528),
    :efficientnet_b7 => (2.0, 3.1, 600),
    :efficientnet_b8 => (2.2, 3.6, 672),
    :efficientnet_l2 => (4.3, 5.3, 800),
)

function get_efficient_net_config(model_name; nclasses=1000, include_top=true, kwargs...)
    block_params =
        NamedTuple{(
            :repeat,
            :kernel,
            :stride,
            :expansion_ratio,
            :in_channels,
            :out_channels,
            :se_ratio,
            :skip_connection,
        )}.([
            (1, (3, 3), 1, 1, 32, 16, 0.25, true),
            (2, (3, 3), 2, 6, 16, 24, 0.25, true),
            (2, (5, 5), 2, 6, 24, 40, 0.25, true),
            (3, (3, 3), 2, 6, 40, 80, 0.25, true),
            (3, (5, 5), 1, 6, 80, 112, 0.25, true),
            (4, (5, 5), 2, 6, 112, 192, 0.25, true),
            (1, (3, 3), 1, 6, 192, 320, 0.25, true),
        ])

    width_coefficient, depth_coefficient, _ = EFFICIENTNET_CONFIG[model_name]

    global_params = (;
        width_coefficient,
        depth_coefficient,
        nclasses,
        depth_divisor=8,
        min_depth=nothing,
        include_top,
    )

    return block_params, global_params
end

function efficient_net_round_filter(filters, width_coefficient, depth_divisor, min_depth)
    width_coefficient ≈ 1 && return filters

    filters *= width_coefficient
    min_depth = min_depth === nothing ? depth_divisor : min_depth

    new_filters = max(
        min_depth, (floor(filters + depth_divisor / 2) ÷ depth_divisor) * depth_divisor
    )
    new_filters < 0.9 * filters && (new_filters += depth_divisor)
    return Int(new_filters)
end

const EFFICIENTNET_PRETRAINED_BASE_URL = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/"

const EFFICIENTNET_PRETRAINED_URLS = Dict(
    :efficientnet_b0 => "efficientnet-b0-355c32eb.pth",
    :efficientnet_b1 => "efficientnet-b1-f1951068.pth",
    :efficientnet_b2 => "efficientnet-b2-8bb594d6.pth",
    :efficientnet_b3 => "efficientnet-b3-5fb5a3c3.pth",
    :efficientnet_b4 => "efficientnet-b4-6ed6700e.pth",
    :efficientnet_b5 => "efficientnet-b5-b6417697.pth",
    :efficientnet_b6 => "efficientnet-b6-c76e70fd.pth",
    :efficientnet_b7 => "efficientnet-b7-dcc49843.pth",
)

function InitializeModels.get_pretrained_weights_url(model::EfficientNet)
    return EFFICIENTNET_PRETRAINED_BASE_URL *
           EFFICIENTNET_PRETRAINED_URLS[model.pretrained_name]
end

# Load the unpickled pytorch weights into the Lux model
function InitializeModels.load_parameters(rng::AbstractRNG, model::EfficientNet, ps_pytorch)
    @set! model.pretrained = false
    ps = LuxCore.initialparameters(rng, model)

    # stem
    PytorchLoadUtils.rebuild_conv_filter!(
        ps.stem[1].weight, ps_pytorch["_conv_stem.weight"]
    )
    copyto!(ps.stem[2].scale, ps_pytorch["_bn0.weight"])
    copyto!(ps.stem[2].bias, ps_pytorch["_bn0.bias"])

    # blocks
    for i in 1:length(model.layer.blocks)
        ps_base = ps.blocks[i]
        base = "_blocks.$(i - 1)"

        if !isempty(ps_base.expansion)
            PytorchLoadUtils.rebuild_conv_filter!(
                ps_base.expansion[1].weight, ps_pytorch[base * "._expand_conv.weight"]
            )
            copyto!(ps_base.expansion[2].scale, ps_pytorch[base * "._bn0.weight"])
            copyto!(ps_base.expansion[2].bias, ps_pytorch[base * "._bn0.bias"])
        end

        PytorchLoadUtils.rebuild_conv_filter!(
            ps_base.depthwise[1].weight, ps_pytorch[base * "._depthwise_conv.weight"]
        )
        copyto!(ps_base.depthwise[2].scale, ps_pytorch[base * "._bn1.weight"])
        copyto!(ps_base.depthwise[2].bias, ps_pytorch[base * "._bn1.bias"])

        if !isempty(ps_base.excitation)
            PytorchLoadUtils.rebuild_conv_filter!(
                ps_base.excitation[2].weight, ps_pytorch[base * "._se_reduce.weight"]
            )
            copyto!(ps_base.excitation[2].bias, ps_pytorch[base * "._se_reduce.bias"])

            PytorchLoadUtils.rebuild_conv_filter!(
                ps_base.excitation[3].weight, ps_pytorch[base * "._se_expand.weight"]
            )
            copyto!(ps_base.excitation[3].bias, ps_pytorch[base * "._se_expand.bias"])
        end

        PytorchLoadUtils.rebuild_conv_filter!(
            ps_base.projection[1].weight, ps_pytorch[base * "._project_conv.weight"]
        )
        copyto!(ps_base.projection[2].scale, ps_pytorch[base * "._bn2.weight"])
        copyto!(ps_base.projection[2].bias, ps_pytorch[base * "._bn2.bias"])
    end

    # head
    if !isempty(ps.head)
        PytorchLoadUtils.rebuild_conv_filter!(
            ps.head[1].weight, ps_pytorch["_conv_head.weight"]
        )
        copyto!(ps.head[2].scale, ps_pytorch["_bn1.weight"])
        copyto!(ps.head[2].bias, ps_pytorch["_bn1.bias"])
    end

    # top
    if !isempty(ps.top)
        copyto!(ps.top.weight, ps_pytorch["_fc.weight"])
        copyto!(ps.top.bias, ps_pytorch["_fc.bias"])
    end

    return InitializeModels.load_parameters_fallback(ps)
end

function InitializeModels.load_states(rng::AbstractRNG, model::EfficientNet, st_pytorch)
    @set! model.pretrained = false
    st = LuxCore.initialstates(rng, model)

    # stem
    copyto!(st.stem[2].running_mean, st_pytorch["_bn0.running_mean"])
    copyto!(st.stem[2].running_var, st_pytorch["_bn0.running_var"])

    # blocks
    for i in 1:length(model.layer.blocks)
        base = "_blocks.$(i - 1)"
        st_base = st.blocks[i]
        if !isempty(st_base.expansion)
            copyto!(
                st_base.expansion[2].running_mean, st_pytorch[base * "._bn0.running_mean"]
            )
            copyto!(
                st_base.expansion[2].running_var, st_pytorch[base * "._bn0.running_var"]
            )
        end

        copyto!(st_base.depthwise[2].running_mean, st_pytorch[base * "._bn1.running_mean"])
        copyto!(st_base.depthwise[2].running_var, st_pytorch[base * "._bn1.running_var"])

        copyto!(st_base.projection[2].running_mean, st_pytorch[base * "._bn2.running_mean"])
        copyto!(st_base.projection[2].running_var, st_pytorch[base * "._bn2.running_var"])
    end

    # head
    if !isempty(st.head)
        copyto!(st.head[2].running_mean, st_pytorch["_bn1.running_mean"])
        copyto!(st.head[2].running_var, st_pytorch["_bn1.running_var"])
    end

    return InitializeModels.load_states_fallback(st)
end
