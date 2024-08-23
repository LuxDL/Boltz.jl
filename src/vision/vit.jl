function __patch_embedding(
        imsize::Dims{2}=(224, 224); in_channels=3, patch_size::Dims{2}=(16, 16),
        embed_planes=768, norm_layer=Returns(Lux.NoOpLayer()), flatten=true)
    im_width, im_height = imsize
    patch_width, patch_height = patch_size

    @argcheck (im_width % patch_width == 0) && (im_height % patch_height == 0)

    return Lux.Chain(Lux.Conv(patch_size, in_channels => embed_planes; stride=patch_size),
        ifelse(flatten, flatten_spatial, identity), norm_layer(embed_planes))
end

# ViT Implementation
function VisionTransformer(;
        imsize::Dims{2}=(256, 256), in_channels::Int=3, patch_size::Dims{2}=(16, 16),
        embed_planes::Int=768, depth::Int=6, number_heads=16,
        mlp_ratio=4.0f0, dropout_rate=0.1f0, embedding_dropout_rate=0.1f0,
        pool::Symbol=:class, num_classes::Int=1000, kwargs...)
    @argcheck pool in (:class, :mean)
    number_patches = prod(imsize .÷ patch_size)

    return Lux.Chain(
        Lux.Chain(__patch_embedding(imsize; in_channels, patch_size, embed_planes),
            Layers.ClassTokens(embed_planes),
            Layers.ViPosEmbedding(embed_planes, number_patches + 1),
            Lux.Dropout(embedding_dropout_rate),
            Layers.VisionTransformerEncoder(
                embed_planes, depth, number_heads; mlp_ratio, dropout_rate),
            Lux.WrappedFunction(ifelse(pool === :class, x -> x[:, 1, :], second_dim_mean));
            disable_optimizations=true),
        Lux.Chain(Lux.LayerNorm((embed_planes,); affine=true),
            Lux.Dense(embed_planes, num_classes, tanh); disable_optimizations=true);
        disable_optimizations=true)
end

#! format: off
const VIT_CONFIGS = Dict(
    :tiny     => (depth=12, embed_planes=0192, number_heads=3                    ),
    :small    => (depth=12, embed_planes=0384, number_heads=6                    ),
    :base     => (depth=12, embed_planes=0768, number_heads=12                   ),
    :large    => (depth=24, embed_planes=1024, number_heads=16                   ),
    :huge     => (depth=32, embed_planes=1280, number_heads=16                   ),
    :giant    => (depth=40, embed_planes=1408, number_heads=16, mlp_ratio=48 / 11),
    :gigantic => (depth=48, embed_planes=1664, number_heads=16, mlp_ratio=64 / 13)
)
#! format: on

"""
    VisionTransformer(name::Symbol; kwargs...)

Creates a Vision Transformer model with the specified configuration.

## Arguments

  - `name::Symbol`: name of the Vision Transformer model to create. The following models are
    available:

## Keyword Arguments

$(INITIALIZE_KWARGS)
"""
function VisionTransformer(name::Symbol; kwargs...)
    @argcheck name in keys(VIT_CONFIGS)
    model = VisionTransformer(; VIT_CONFIGS[name]..., kwargs...)
    return __maybe_initialize_model(name, model; kwargs...)
end

const ViT = VisionTransformer
