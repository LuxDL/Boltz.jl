import{_ as a,c as n,a2 as p,o as l}from"./chunks/framework.DnVQYp8_.js";const k=JSON.parse('{"title":"Getting Started","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/1_GettingStarted.md","filePath":"tutorials/1_GettingStarted.md","lastUpdated":null}'),e={name:"tutorials/1_GettingStarted.md"};function i(t,s,r,c,o,h){return l(),n("div",null,s[0]||(s[0]=[p(`<h1 id="Getting-Started" tabindex="-1">Getting Started <a class="header-anchor" href="#Getting-Started" aria-label="Permalink to &quot;Getting Started {#Getting-Started}&quot;">​</a></h1><div class="tip custom-block"><p class="custom-block-title">Prerequisites</p><p>Here we assume that you are familiar with <a href="https://lux.csail.mit.edu/stable/" target="_blank" rel="noreferrer"><code>Lux.jl</code></a>. If not please take a look at the <a href="https://lux.csail.mit.edu/stable/tutorials/" target="_blank" rel="noreferrer">Lux.jl tutoials</a>.</p></div><p><code>Boltz.jl</code> is just like <code>Lux.jl</code> but comes with more &quot;batteries included&quot;. Let&#39;s start by defining an MLP model.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, Boltz, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>    562.9 ms  ✓ GPUArraysCore</span></span>
<span class="line"><span>   1054.2 ms  ✓ Functors</span></span>
<span class="line"><span>    790.2 ms  ✓ ArrayInterface → ArrayInterfaceGPUArraysCoreExt</span></span>
<span class="line"><span>   1540.5 ms  ✓ LuxCore</span></span>
<span class="line"><span>   1403.3 ms  ✓ MLDataDevices</span></span>
<span class="line"><span>   1076.5 ms  ✓ LuxCore → LuxCoreChainRulesCoreExt</span></span>
<span class="line"><span>   1437.0 ms  ✓ Optimisers</span></span>
<span class="line"><span>    974.9 ms  ✓ LuxCore → LuxCoreFunctorsExt</span></span>
<span class="line"><span>    806.7 ms  ✓ LuxCore → LuxCoreEnzymeCoreExt</span></span>
<span class="line"><span>    630.9 ms  ✓ LuxCore → LuxCoreSetfieldExt</span></span>
<span class="line"><span>    564.3 ms  ✓ LuxCore → LuxCoreMLDataDevicesExt</span></span>
<span class="line"><span>    760.2 ms  ✓ MLDataDevices → MLDataDevicesChainRulesCoreExt</span></span>
<span class="line"><span>   4019.8 ms  ✓ WeightInitializers</span></span>
<span class="line"><span>    828.8 ms  ✓ WeightInitializers → WeightInitializersChainRulesCoreExt</span></span>
<span class="line"><span>   6352.6 ms  ✓ NNlib</span></span>
<span class="line"><span>    763.9 ms  ✓ NNlib → NNlibEnzymeCoreExt</span></span>
<span class="line"><span>    827.1 ms  ✓ NNlib → NNlibForwardDiffExt</span></span>
<span class="line"><span>   5087.3 ms  ✓ LuxLib</span></span>
<span class="line"><span>   8499.9 ms  ✓ Lux</span></span>
<span class="line"><span>  19 dependencies successfully precompiled in 22 seconds. 88 already precompiled.</span></span>
<span class="line"><span>Precompiling Boltz...</span></span>
<span class="line"><span>   4976.3 ms  ✓ Boltz</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 5 seconds. 119 already precompiled.</span></span></code></pre></div><h2 id="Multi-Layer-Perceptron" tabindex="-1">Multi-Layer Perceptron <a class="header-anchor" href="#Multi-Layer-Perceptron" aria-label="Permalink to &quot;Multi-Layer Perceptron {#Multi-Layer-Perceptron}&quot;">​</a></h2><p>If we were to do this in <code>Lux.jl</code> we would write the following:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">784</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Chain(</span></span>
<span class="line"><span>    layer_1 = Dense(784 =&gt; 256, relu),  # 200_960 parameters</span></span>
<span class="line"><span>    layer_2 = Dense(256 =&gt; 10),         # 2_570 parameters</span></span>
<span class="line"><span>)         # Total: 203_530 parameters,</span></span>
<span class="line"><span>          #        plus 0 states.</span></span></code></pre></div><p>But in <code>Boltz.jl</code> we can do this:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">MLP</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">784</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), relu)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>MLP(</span></span>
<span class="line"><span>    chain = Chain(</span></span>
<span class="line"><span>        block1 = DenseNormActDropoutBlock(</span></span>
<span class="line"><span>            block = Chain(</span></span>
<span class="line"><span>                dense = Dense(784 =&gt; 256, relu),  # 200_960 parameters</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>        block2 = DenseNormActDropoutBlock(</span></span>
<span class="line"><span>            block = Chain(</span></span>
<span class="line"><span>                dense = Dense(256 =&gt; 10),  # 2_570 parameters</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>    ),</span></span>
<span class="line"><span>)         # Total: 203_530 parameters,</span></span>
<span class="line"><span>          #        plus 0 states.</span></span></code></pre></div><p>The <code>MLP</code> function is just a convenience wrapper around <code>Lux.Chain</code> that constructs a multi-layer perceptron with the given number of layers and activation function.</p><h2 id="How-about-VGG?" tabindex="-1">How about VGG? <a class="header-anchor" href="#How-about-VGG?" aria-label="Permalink to &quot;How about VGG? {#How-about-VGG?}&quot;">​</a></h2><p>Let&#39;s take a look at the <code>Vision</code> module. We can construct a VGG model with the following code:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Vision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">VGG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">13</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>VGG(</span></span>
<span class="line"><span>    layer = Chain(</span></span>
<span class="line"><span>        feature_extractor = VGGFeatureExtractor(</span></span>
<span class="line"><span>            model = Chain(</span></span>
<span class="line"><span>                layer_1 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 3 =&gt; 64, relu, pad=1),  # 1_792 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 64 =&gt; 64, relu, pad=1),  # 36_928 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_2 = MaxPool((2, 2)),</span></span>
<span class="line"><span>                layer_3 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 64 =&gt; 128, relu, pad=1),  # 73_856 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 128 =&gt; 128, relu, pad=1),  # 147_584 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_4 = MaxPool((2, 2)),</span></span>
<span class="line"><span>                layer_5 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 128 =&gt; 256, relu, pad=1),  # 295_168 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 256 =&gt; 256, relu, pad=1),  # 590_080 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_6 = MaxPool((2, 2)),</span></span>
<span class="line"><span>                layer_7 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 256 =&gt; 512, relu, pad=1),  # 1_180_160 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 512 =&gt; 512, relu, pad=1),  # 2_359_808 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_8 = MaxPool((2, 2)),</span></span>
<span class="line"><span>                layer_9 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 512 =&gt; 512, relu, pad=1),  # 2_359_808 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 512 =&gt; 512, relu, pad=1),  # 2_359_808 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_10 = MaxPool((2, 2)),</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>        classifier = VGGClassifier(</span></span>
<span class="line"><span>            model = Chain(</span></span>
<span class="line"><span>                layer_1 = Lux.FlattenLayer{Nothing}(nothing),</span></span>
<span class="line"><span>                layer_2 = Dense(25088 =&gt; 4096, relu),  # 102_764_544 parameters</span></span>
<span class="line"><span>                layer_3 = Dropout(0.5),</span></span>
<span class="line"><span>                layer_4 = Dense(4096 =&gt; 4096, relu),  # 16_781_312 parameters</span></span>
<span class="line"><span>                layer_5 = Dropout(0.5),</span></span>
<span class="line"><span>                layer_6 = Dense(4096 =&gt; 1000),  # 4_097_000 parameters</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>    ),</span></span>
<span class="line"><span>)         # Total: 133_047_848 parameters,</span></span>
<span class="line"><span>          #        plus 4 states.</span></span></code></pre></div><p>We can also load pretrained ImageNet weights using</p><div class="tip custom-block"><p class="custom-block-title">Load JLD2</p><p>You need to load <code>JLD2</code> before being able to load pretrained weights.</p></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> JLD2</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Vision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">VGG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">13</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>VGG(</span></span>
<span class="line"><span>    layer = Chain(</span></span>
<span class="line"><span>        feature_extractor = VGGFeatureExtractor(</span></span>
<span class="line"><span>            model = Chain(</span></span>
<span class="line"><span>                layer_1 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 3 =&gt; 64, relu, pad=1),  # 1_792 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 64 =&gt; 64, relu, pad=1),  # 36_928 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_2 = MaxPool((2, 2)),</span></span>
<span class="line"><span>                layer_3 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 64 =&gt; 128, relu, pad=1),  # 73_856 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 128 =&gt; 128, relu, pad=1),  # 147_584 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_4 = MaxPool((2, 2)),</span></span>
<span class="line"><span>                layer_5 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 128 =&gt; 256, relu, pad=1),  # 295_168 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 256 =&gt; 256, relu, pad=1),  # 590_080 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_6 = MaxPool((2, 2)),</span></span>
<span class="line"><span>                layer_7 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 256 =&gt; 512, relu, pad=1),  # 1_180_160 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 512 =&gt; 512, relu, pad=1),  # 2_359_808 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_8 = MaxPool((2, 2)),</span></span>
<span class="line"><span>                layer_9 = ConvNormActivation(</span></span>
<span class="line"><span>                    model = Chain(</span></span>
<span class="line"><span>                        block1 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 512 =&gt; 512, relu, pad=1),  # 2_359_808 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                        block2 = ConvNormActivationBlock(</span></span>
<span class="line"><span>                            block = Conv((3, 3), 512 =&gt; 512, relu, pad=1),  # 2_359_808 parameters</span></span>
<span class="line"><span>                        ),</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_10 = MaxPool((2, 2)),</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>        classifier = VGGClassifier(</span></span>
<span class="line"><span>            model = Chain(</span></span>
<span class="line"><span>                layer_1 = Lux.FlattenLayer{Nothing}(nothing),</span></span>
<span class="line"><span>                layer_2 = Dense(25088 =&gt; 4096, relu),  # 102_764_544 parameters</span></span>
<span class="line"><span>                layer_3 = Dropout(0.5),</span></span>
<span class="line"><span>                layer_4 = Dense(4096 =&gt; 4096, relu),  # 16_781_312 parameters</span></span>
<span class="line"><span>                layer_5 = Dropout(0.5),</span></span>
<span class="line"><span>                layer_6 = Dense(4096 =&gt; 1000),  # 4_097_000 parameters</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>    ),</span></span>
<span class="line"><span>)         # Total: 133_047_848 parameters,</span></span>
<span class="line"><span>          #        plus 4 states.</span></span></code></pre></div><h2 id="Loading-Models-from-Metalhead-(Flux.jl)" tabindex="-1">Loading Models from Metalhead (Flux.jl) <a class="header-anchor" href="#Loading-Models-from-Metalhead-(Flux.jl)" aria-label="Permalink to &quot;Loading Models from Metalhead (Flux.jl) {#Loading-Models-from-Metalhead-(Flux.jl)}&quot;">​</a></h2><p>We can load models from Metalhead (Flux.jl), just remember to load <code>Metalhead</code> before.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Metalhead</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Vision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ResNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">18</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>MetalheadWrapperLayer(</span></span>
<span class="line"><span>    layer = Chain(</span></span>
<span class="line"><span>        layer_1 = Chain(</span></span>
<span class="line"><span>            layer_1 = Chain(</span></span>
<span class="line"><span>                layer_1 = Conv((7, 7), 3 =&gt; 64, pad=3, stride=2, use_bias=false),  # 9_408 parameters</span></span>
<span class="line"><span>                layer_2 = BatchNorm(64, relu, affine=true, track_stats=true),  # 128 parameters, plus 129</span></span>
<span class="line"><span>                layer_3 = MaxPool((3, 3), pad=1, stride=2),</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>            layer_2 = Chain(</span></span>
<span class="line"><span>                layer_1 = Parallel(</span></span>
<span class="line"><span>                    connection = addact(NNlib.relu, ...),</span></span>
<span class="line"><span>                    layer_1 = Lux.NoOpLayer(),</span></span>
<span class="line"><span>                    layer_2 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((3, 3), 64 =&gt; 64, pad=1, use_bias=false),  # 36_864 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(64, affine=true, track_stats=true),  # 128 parameters, plus 129</span></span>
<span class="line"><span>                        layer_3 = WrappedFunction(relu),</span></span>
<span class="line"><span>                        layer_4 = Conv((3, 3), 64 =&gt; 64, pad=1, use_bias=false),  # 36_864 parameters</span></span>
<span class="line"><span>                        layer_5 = BatchNorm(64, affine=true, track_stats=true),  # 128 parameters, plus 129</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_2 = Parallel(</span></span>
<span class="line"><span>                    connection = addact(NNlib.relu, ...),</span></span>
<span class="line"><span>                    layer_1 = Lux.NoOpLayer(),</span></span>
<span class="line"><span>                    layer_2 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((3, 3), 64 =&gt; 64, pad=1, use_bias=false),  # 36_864 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(64, affine=true, track_stats=true),  # 128 parameters, plus 129</span></span>
<span class="line"><span>                        layer_3 = WrappedFunction(relu),</span></span>
<span class="line"><span>                        layer_4 = Conv((3, 3), 64 =&gt; 64, pad=1, use_bias=false),  # 36_864 parameters</span></span>
<span class="line"><span>                        layer_5 = BatchNorm(64, affine=true, track_stats=true),  # 128 parameters, plus 129</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>            layer_3 = Chain(</span></span>
<span class="line"><span>                layer_1 = Parallel(</span></span>
<span class="line"><span>                    connection = addact(NNlib.relu, ...),</span></span>
<span class="line"><span>                    layer_1 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((1, 1), 64 =&gt; 128, stride=2, use_bias=false),  # 8_192 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(128, affine=true, track_stats=true),  # 256 parameters, plus 257</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                    layer_2 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((3, 3), 64 =&gt; 128, pad=1, stride=2, use_bias=false),  # 73_728 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(128, affine=true, track_stats=true),  # 256 parameters, plus 257</span></span>
<span class="line"><span>                        layer_3 = WrappedFunction(relu),</span></span>
<span class="line"><span>                        layer_4 = Conv((3, 3), 128 =&gt; 128, pad=1, use_bias=false),  # 147_456 parameters</span></span>
<span class="line"><span>                        layer_5 = BatchNorm(128, affine=true, track_stats=true),  # 256 parameters, plus 257</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_2 = Parallel(</span></span>
<span class="line"><span>                    connection = addact(NNlib.relu, ...),</span></span>
<span class="line"><span>                    layer_1 = Lux.NoOpLayer(),</span></span>
<span class="line"><span>                    layer_2 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((3, 3), 128 =&gt; 128, pad=1, use_bias=false),  # 147_456 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(128, affine=true, track_stats=true),  # 256 parameters, plus 257</span></span>
<span class="line"><span>                        layer_3 = WrappedFunction(relu),</span></span>
<span class="line"><span>                        layer_4 = Conv((3, 3), 128 =&gt; 128, pad=1, use_bias=false),  # 147_456 parameters</span></span>
<span class="line"><span>                        layer_5 = BatchNorm(128, affine=true, track_stats=true),  # 256 parameters, plus 257</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>            layer_4 = Chain(</span></span>
<span class="line"><span>                layer_1 = Parallel(</span></span>
<span class="line"><span>                    connection = addact(NNlib.relu, ...),</span></span>
<span class="line"><span>                    layer_1 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((1, 1), 128 =&gt; 256, stride=2, use_bias=false),  # 32_768 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(256, affine=true, track_stats=true),  # 512 parameters, plus 513</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                    layer_2 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((3, 3), 128 =&gt; 256, pad=1, stride=2, use_bias=false),  # 294_912 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(256, affine=true, track_stats=true),  # 512 parameters, plus 513</span></span>
<span class="line"><span>                        layer_3 = WrappedFunction(relu),</span></span>
<span class="line"><span>                        layer_4 = Conv((3, 3), 256 =&gt; 256, pad=1, use_bias=false),  # 589_824 parameters</span></span>
<span class="line"><span>                        layer_5 = BatchNorm(256, affine=true, track_stats=true),  # 512 parameters, plus 513</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_2 = Parallel(</span></span>
<span class="line"><span>                    connection = addact(NNlib.relu, ...),</span></span>
<span class="line"><span>                    layer_1 = Lux.NoOpLayer(),</span></span>
<span class="line"><span>                    layer_2 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((3, 3), 256 =&gt; 256, pad=1, use_bias=false),  # 589_824 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(256, affine=true, track_stats=true),  # 512 parameters, plus 513</span></span>
<span class="line"><span>                        layer_3 = WrappedFunction(relu),</span></span>
<span class="line"><span>                        layer_4 = Conv((3, 3), 256 =&gt; 256, pad=1, use_bias=false),  # 589_824 parameters</span></span>
<span class="line"><span>                        layer_5 = BatchNorm(256, affine=true, track_stats=true),  # 512 parameters, plus 513</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>            layer_5 = Chain(</span></span>
<span class="line"><span>                layer_1 = Parallel(</span></span>
<span class="line"><span>                    connection = addact(NNlib.relu, ...),</span></span>
<span class="line"><span>                    layer_1 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((1, 1), 256 =&gt; 512, stride=2, use_bias=false),  # 131_072 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(512, affine=true, track_stats=true),  # 1_024 parameters, plus 1_025</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                    layer_2 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((3, 3), 256 =&gt; 512, pad=1, stride=2, use_bias=false),  # 1_179_648 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(512, affine=true, track_stats=true),  # 1_024 parameters, plus 1_025</span></span>
<span class="line"><span>                        layer_3 = WrappedFunction(relu),</span></span>
<span class="line"><span>                        layer_4 = Conv((3, 3), 512 =&gt; 512, pad=1, use_bias=false),  # 2_359_296 parameters</span></span>
<span class="line"><span>                        layer_5 = BatchNorm(512, affine=true, track_stats=true),  # 1_024 parameters, plus 1_025</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>                layer_2 = Parallel(</span></span>
<span class="line"><span>                    connection = addact(NNlib.relu, ...),</span></span>
<span class="line"><span>                    layer_1 = Lux.NoOpLayer(),</span></span>
<span class="line"><span>                    layer_2 = Chain(</span></span>
<span class="line"><span>                        layer_1 = Conv((3, 3), 512 =&gt; 512, pad=1, use_bias=false),  # 2_359_296 parameters</span></span>
<span class="line"><span>                        layer_2 = BatchNorm(512, affine=true, track_stats=true),  # 1_024 parameters, plus 1_025</span></span>
<span class="line"><span>                        layer_3 = WrappedFunction(relu),</span></span>
<span class="line"><span>                        layer_4 = Conv((3, 3), 512 =&gt; 512, pad=1, use_bias=false),  # 2_359_296 parameters</span></span>
<span class="line"><span>                        layer_5 = BatchNorm(512, affine=true, track_stats=true),  # 1_024 parameters, plus 1_025</span></span>
<span class="line"><span>                    ),</span></span>
<span class="line"><span>                ),</span></span>
<span class="line"><span>            ),</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>        layer_2 = Chain(</span></span>
<span class="line"><span>            layer_1 = AdaptiveMeanPool((1, 1)),</span></span>
<span class="line"><span>            layer_2 = WrappedFunction(flatten),</span></span>
<span class="line"><span>            layer_3 = Dense(512 =&gt; 1000),  # 513_000 parameters</span></span>
<span class="line"><span>        ),</span></span>
<span class="line"><span>    ),</span></span>
<span class="line"><span>)         # Total: 11_689_512 parameters,</span></span>
<span class="line"><span>          #        plus 9_620 states.</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MLDataDevices)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.2</span></span>
<span class="line"><span>Commit 5e9a32e7af2 (2024-12-01 20:02 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 4 × AMD EPYC 7763 64-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)</span></span>
<span class="line"><span>Threads: 1 default, 0 interactive, 1 GC (on 4 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 1</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,30)]))}const u=a(e,[["render",i]]);export{k as __pageData,u as default};
