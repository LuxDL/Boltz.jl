struct BlockParams
    repeat::Int
    kernel::Tuple{Int,Int}
    stride::Int
    expand_ratio::Int
    in_channels::Int
    out_channels::Int
    se_ratio::Real
    skip_connection::Bool
end

struct GlobalParams
    width_coef::Real
    depth_coef::Real
    image_size::Dims{2}

    n_classes::Int

    depth_divisor::Int
    min_depth::Union{Nothing,Int}
    include_top::Bool
end

# (width_coefficient, depth_coefficient, resolution)
const EFFICIENTNET_CONFIG = Dict(
    "efficientnet-b0" => (1.0, 1.0, 224),
    "efficientnet-b1" => (1.0, 1.1, 240),
    "efficientnet-b2" => (1.1, 1.2, 260),
    "efficientnet-b3" => (1.2, 1.4, 300),
    "efficientnet-b4" => (1.4, 1.8, 380),
    "efficientnet-b5" => (1.6, 2.2, 456),
    "efficientnet-b6" => (1.8, 2.6, 528),
    "efficientnet-b7" => (2.0, 3.1, 600),
    "efficientnet-b8" => (2.2, 3.6, 672),
    "efficientnet-l2" => (4.3, 5.3, 800),
)

function get_model_params(model_name; n_classes=1000, include_top=true, kwargs...)
    block_params = [
        BlockParams(1, (3, 3), 1, 1, 32, 16, 0.25, true),
        BlockParams(2, (3, 3), 2, 6, 16, 24, 0.25, true),
        BlockParams(2, (5, 5), 2, 6, 24, 40, 0.25, true),
        BlockParams(3, (3, 3), 2, 6, 40, 80, 0.25, true),
        BlockParams(3, (5, 5), 1, 6, 80, 112, 0.25, true),
        BlockParams(4, (5, 5), 2, 6, 112, 192, 0.25, true),
        BlockParams(1, (3, 3), 1, 6, 192, 320, 0.25, true),
    ]

    wc, dc, res = EFFICIENTNET_CONFIG[model_name]
    global_params = GlobalParams(wc, dc, (res, res), n_classes, 8, nothing, include_top)
    return block_params, global_params
end

function round_filter(filters, global_params::GlobalParams)
    global_params.width_coef ≈ 1 && return filters

    depth_divisor = global_params.depth_divisor
    filters *= global_params.width_coef
    min_depth = global_params.min_depth
    min_depth = min_depth ≡ nothing ? depth_divisor : min_depth

    new_filters = max(
        min_depth, (floor(filters + depth_divisor / 2) ÷ depth_divisor) * depth_divisor
    )
    new_filters < 0.9 * filters && (new_filters += global_params.depth_divisor)
    return new_filters
end

@concrete struct MBConv <: AbstractLuxContainerLayer{(
    :expansion, :depthwise, :excitation, :projection
)}
    expansion
    depthwise
    excitation
    projection
    do_skip::Bool
end

"""
Mobile Inverted Residual Bottleneck Block.

Args:
    in_channels: Number of input channels.
    out_channels: Number of output channels.
    expansion_ratio:
        Expansion ratio defines the number of output channels.
        Set to `1` to disable expansion phase.
        `out_channels = input_channels * expansion_ratio`.
    kernel: Size of the kernel for the depthwise conv phase.
    stride: Size of the stride for the depthwise conv phase.
    se_ratio:
        Squeeze-Excitation ratio. Should be in `(0, 1]` range.
        Set to `-1` to disable.
    skip_connection: Whether to use skip connection and drop connect.
"""
function MBConv(
    in_channels, out_channels, kernel, stride; expansion_ratio, se_ratio, skip_connection
)
    @assert 0 < se_ratio ≤ 1

    do_skip = skip_connection && stride == 1 && in_channels == out_channels
    do_expansion = expansion_ratio != 1
    pad, use_bias = SamePad(), false

    mid_channels = ceil(Int, in_channels * expansion_ratio)
    expansion = if do_expansion
        Chain(
            Conv((1, 1), in_channels => mid_channels; use_bias, pad),
            BatchNorm(mid_channels, swish),
        )
    else
        NoOpLayer()
    end

    depthwise = Chain(
        Conv(
            kernel, mid_channels => mid_channels; use_bias, stride, pad, groups=mid_channels
        ),
        BatchNorm(mid_channels, swish),
    )

    n_squeezed_channels = max(1, ceil(Int, in_channels * se_ratio))
    excitation = Chain(
        AdaptiveMeanPool((1, 1)),
        Conv((1, 1), mid_channels => n_squeezed_channels, swish; pad),
        Conv((1, 1), n_squeezed_channels => mid_channels; pad),
    )
    projection = Chain(
        Conv((1, 1), mid_channels => out_channels; pad, use_bias), BatchNorm(out_channels)
    )

    return MBConv(
        expansion, depthwise, excitation, projection, do_expansion, do_excitation, do_skip
    )
end

function (m::MBConv)(x, ps, st)
    o, st_expansion = m.expansion(x, ps.expansion, st.expansion)
    o, st_depthwise = m.depthwise(o, ps.depthwise, st.depthwise)
    oe, st_excitation = m.excitation(o, ps.excitation, st.excitation)

    o = σ.(oe) .* o
    o, st_projection = m.projection(o, ps.projection, st.projection)

    m.do_skip && (o = o .+ x)

    return (
        o,
        (;
            expansion=st_expansion,
            depthwise=st_depthwise,
            projection=st_projection,
            excitation=st_excitation,
        ),
    )
end

struct EfficientNet{H,F,S,B,P,FL} <: Lux.AbstractLuxLayer
    stem::S
    blocks::B

    head::H
    pooling::P
    top::F

    flatten::FL

    stages::NTuple{4,Int}
    stages_channels::NTuple{5,Int}
    pretrained_name::String
    pretrained::Bool
end

function LuxCore.initialparameters(rng::AbstractRNG, l::EfficientNet)
    ps = (
        stem=Lux.initialparameters(rng, l.stem),
        blocks=Lux.initialparameters(rng, l.blocks),
        head=Lux.initialparameters(rng, l.head),
        pooling=Lux.initialparameters(rng, l.pooling),
        top=Lux.initialparameters(rng, l.top),
        flatten=Lux.initialparameters(rng, l.flatten),
    )
    if l.pretrained
        params = download_params(l, l.pretrained_name)
        _load_params!(l, ps, params)
    end

    return (ps)
end

function LuxCore.initialstates(rng::AbstractRNG, l::EfficientNet)
    st = (
        stem=Lux.initialstates(rng, l.stem),
        blocks=Lux.initialstates(rng, l.blocks),
        head=Lux.initialstates(rng, l.head),
        pooling=Lux.initialstates(rng, l.pooling),
        top=Lux.initialstates(rng, l.top),
        flatten=Lux.initialstates(rng, l.flatten),
    )
    if l.pretrained
        params = download_params(l, l.pretrained_name)
        _load_states!(l, st, params)
    end

    return (st)
end

function EfficientNet(
    model_name,
    block_params,
    global_params;
    pretrained=false,
    include_head=true,
    in_channels=3,
)
    pad, use_bias = SamePad(), false
    out_channels = round_filter(32, global_params)
    stem = Chain(
        Conv((3, 3), in_channels => out_channels; use_bias, stride=2, pad),
        BatchNorm(out_channels, swish),
    )

    blocks = MBConv[]
    for bp in block_params
        in_channels = round_filter(bp.in_channels, global_params)
        out_channels = round_filter(bp.out_channels, global_params)
        repeat = if global_params.depth_coef ≈ 1
            bp.repeat
        else
            ceil(Int64, global_params.depth_coef * bp.repeat)
        end

        expansion_ratio, se_ratio, skip_connection = bp.expand_ratio,
        bp.se_ratio,
        bp.skip_connection
        push!(
            blocks,
            MBConv(
                in_channels,
                out_channels,
                bp.kernel,
                bp.stride;
                expansion_ratio,
                se_ratio,
                skip_connection,
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
                    expansion_ratio,
                    se_ratio,
                    skip_connection,
                ),
            )
        end
    end
    blocks = Chain(blocks...)

    stages = get_stages(model_name)
    channels = stages_channels(model_name)
    flatten = FlattenLayer()
    include_head || return EfficientNet(
        stem, blocks, nothing, nothing, nothing, flatten, stages, channels
    )

    head_out_channels = round_filter(1280, global_params)
    head = Chain(
        Conv((1, 1), out_channels => head_out_channels; use_bias, pad),
        BatchNorm(head_out_channels, swish),
    )

    top = if global_params.include_top
        Dense(head_out_channels, global_params.n_classes)
    else
        nothing
    end
    return EfficientNet(
        stem,
        blocks,
        head,
        AdaptiveMeanPool((1, 1)),
        top,
        flatten,
        stages,
        channels,
        model_name,
        pretrained,
    )
end

function EfficientNet(model_name::String; kwargs...)
    return EfficientNet(model_name, get_model_params(model_name; kwargs...)...; kwargs...)
end

function (m::EfficientNet{Nothing})(x, ps, st)
    o, st_stem = m.stem(x, ps.stem, st.stem)
    o, st_blocks = m.blocks(o, ps.blocks, st.blocks)
    new_st = (;
        stem=st_stem, blocks=st_blocks, head=st.head, pooling=st.pooling, top=st.top
    )
    return (o, new_st)
end

function (m::EfficientNet{H,Nothing})(x, ps, st) where {H}
    o, st_stem = m.stem(x, ps.stem, st.stem)
    o, st_blocks = m.blocks(o, ps.blocks, st.blocks)

    o, st_head = m.head(o, ps.head, st.head)
    o, st_pooling = m.pooling(o, ps.pooling, st.pooling)
    new_st = (;
        stem=st_stem, blocks=st_blocks, head=st_head, pooling=st_pooling, top=st.top
    )
    return (o, new_st)
end

function (m::EfficientNet{H,F})(x, ps, st) where {H,F}
    o, st_stem = m.stem(x, ps.stem, st.stem)
    o, st_blocks = m.blocks(o, ps.blocks, st.blocks)

    o, st_head = m.head(o, ps.head, st.head)
    o, st_pooling = m.pooling(o, ps.pooling, st.pooling)

    o, st_flatten = m.flatten(o, ps.flatten, st.flatten)
    o, st_top = m.top(o, ps.top, st.top)
    new_st = (;
        stem=st_stem,
        blocks=st_blocks,
        head=st_head,
        pooling=st_pooling,
        top=st_top,
        flatten=st_flatten,
    )
    return (o, new_st)
end

function get_stages(model_name)
    return Dict(
        "efficientnet-b0" => (3, 5, 9, 16),
        "efficientnet-b1" => (5, 8, 16, 23),
        "efficientnet-b2" => (5, 8, 16, 23),
        "efficientnet-b3" => (5, 8, 18, 26),
        "efficientnet-b4" => (6, 10, 22, 32),
        "efficientnet-b5" => (8, 13, 27, 39),
        "efficientnet-b6" => (9, 15, 31, 45),
        "efficientnet-b7" => (11, 18, 38, 55),
    )[model_name]
end
function stages_channels(model_name)
    return Dict(
        "efficientnet-b0" => (32, 24, 40, 112, 320),
        "efficientnet-b1" => (32, 24, 40, 112, 320),
        "efficientnet-b2" => (32, 24, 48, 120, 352),
        "efficientnet-b3" => (40, 32, 48, 136, 384),
        "efficientnet-b4" => (48, 32, 56, 160, 448),
        "efficientnet-b5" => (48, 40, 64, 176, 512),
        "efficientnet-b6" => (56, 40, 72, 200, 576),
        "efficientnet-b7" => (64, 48, 80, 224, 640),
    )[model_name]
end

# Pytorch loading utils.
function rebuild_conv!(dst, src)
    shape = dst |> size
    filter_x, filter_y = shape[1:2] .+ 1
    for (i, j, k, m) in Iterators.product([1:s for s in shape]...)
        dst[filter_x - i, filter_y - j, k, m] = src[m, k, j, i]
    end
end

function _load_params_stem!(model::EfficientNet, ps, params)
    rebuild_conv!(ps.stem[1].weight, params["_conv_stem.weight"])
    copyto!(ps.stem[2].scale, params["_bn0.weight"])
    return copyto!(ps.stem[2].bias, params["_bn0.bias"])
end

function _load_states_stem!(model::EfficientNet, st, params)
    copyto!(st.stem[2].running_mean, params["_bn0.running_mean"])
    return copyto!(st.stem[2].running_var, params["_bn0.running_var"])
end

function _load_params_block!(block::MBConv, ps, params, base)
    if !isempty(ps.expansion)
        rebuild_conv!(ps.expansion[1].weight, params[base * "._expand_conv.weight"])
        copyto!(ps.expansion[2].scale, params[base * "._bn0.weight"])
        copyto!(ps.expansion[2].bias, params[base * "._bn0.bias"])
    end

    rebuild_conv!(ps.depthwise[1].weight, params[base * "._depthwise_conv.weight"])
    copyto!(ps.depthwise[2].scale, params[base * "._bn1.weight"])
    copyto!(ps.depthwise[2].bias, params[base * "._bn1.bias"])

    if !isempty(ps.excitation)
        rebuild_conv!(ps.excitation[2].weight, params[base * "._se_reduce.weight"])
        copyto!(ps.excitation[2].bias, params[base * "._se_reduce.bias"])
        rebuild_conv!(ps.excitation[3].weight, params[base * "._se_expand.weight"])
        copyto!(ps.excitation[3].bias, params[base * "._se_expand.bias"])
    end

    rebuild_conv!(ps.projection[1].weight, params[base * "._project_conv.weight"])
    copyto!(ps.projection[2].scale, params[base * "._bn2.weight"])
    return copyto!(ps.projection[2].bias, params[base * "._bn2.bias"])
end

function _load_states_block!(block::MBConv, st, params, base)
    if !isempty(st.expansion)
        copyto!(st.expansion[2].running_mean, params[base * "._bn0.running_mean"])
        copyto!(st.expansion[2].running_var, params[base * "._bn0.running_var"])
    end

    copyto!(st.depthwise[2].running_mean, params[base * "._bn1.running_mean"])
    copyto!(st.depthwise[2].running_var, params[base * "._bn1.running_var"])
    copyto!(st.projection[2].running_mean, params[base * "._bn2.running_mean"])
    return copyto!(st.projection[2].running_var, params[base * "._bn2.running_var"])
end

function _load_params_blocks!(model::EfficientNet, ps, params)
    for i in 1:length(model.blocks)
        _load_params_block!(model.blocks[i], ps.blocks[i], params, "_blocks.$(i - 1)")
    end
end

function _load_states_blocks!(model::EfficientNet, st, params)
    for i in 1:length(model.blocks)
        _load_states_block!(model.blocks[i], st.blocks[i], params, "_blocks.$(i - 1)")
    end
end

function _load_params_head!(model::EfficientNet, ps, params)
    if model.head ≢ nothing
        rebuild_conv!(ps.head[1].weight, params["_conv_head.weight"])
        copyto!(ps.head[2].scale, params["_bn1.weight"])
        copyto!(ps.head[2].bias, params["_bn1.bias"])
    end

    if model.top ≢ nothing
        copyto!(ps.top.weight, params["_fc.weight"])
        copyto!(ps.top.bias, params["_fc.bias"])
    end
end

function _load_states_head!(model::EfficientNet, st, params)
    if model.head ≢ nothing
        copyto!(st.head[2].running_mean, params["_bn1.running_mean"])
        copyto!(st.head[2].running_var, params["_bn1.running_var"])
    end
end

@inline function _load_params!(model::EfficientNet, ps, params)
    _load_params_stem!(model, ps, params)
    _load_params_blocks!(model, ps, params)
    return _load_params_head!(model, ps, params)
end

@inline function _load_states!(model::EfficientNet, st, params)
    _load_states_stem!(model, st, params)
    _load_states_blocks!(model, st, params)
    return _load_states_head!(model, st, params)
end

function download_params(model::EfficientNet, model_name; cache_dir=nothing)
    url_base = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/"
    params_file = Dict(
        "efficientnet-b0" => "efficientnet-b0-355c32eb.pth",
        "efficientnet-b1" => "efficientnet-b1-f1951068.pth",
        "efficientnet-b2" => "efficientnet-b2-8bb594d6.pth",
        "efficientnet-b3" => "efficientnet-b3-5fb5a3c3.pth",
        "efficientnet-b4" => "efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5" => "efficientnet-b5-b6417697.pth",
        "efficientnet-b6" => "efficientnet-b6-c76e70fd.pth",
        "efficientnet-b7" => "efficientnet-b7-dcc49843.pth",
    )[model_name]

    if cache_dir ≡ nothing
        cache_dir = joinpath(tempdir(), "EfficientNet.jl", ".cache")
        !isdir(cache_dir) && mkpath(cache_dir)
        @info "Using default cache dir $cache_dir"
    end

    params_path = joinpath(cache_dir, params_file)
    if !isfile(params_path)
        download_url = url_base * params_file
        @info "Downloading $model_name params:\n\t- from URL: $download_url \n\t- to directory: $params_path"
        download(download_url, params_path)
        @info "Finished downloading params."
    end

    return params = Pickle.Torch.THload(params_path)
end
