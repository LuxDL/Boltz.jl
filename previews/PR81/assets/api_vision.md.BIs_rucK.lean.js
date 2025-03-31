import{_ as n,c as o,j as t,a as i,G as l,a2 as s,B as d,o as r}from"./chunks/framework.Ck_s-khZ.js";const A=JSON.parse('{"title":"Computer Vision Models (Vision API)","description":"","frontmatter":{},"headers":[],"relativePath":"api/vision.md","filePath":"api/vision.md","lastUpdated":null}'),p={name:"api/vision.md"},h={class:"jldocstring custom-block",open:""},c={class:"jldocstring custom-block",open:""},g={class:"jldocstring custom-block",open:""},k={class:"jldocstring custom-block",open:""},u={class:"jldocstring custom-block",open:""},y={class:"jldocstring custom-block",open:""},m={class:"jldocstring custom-block",open:""},b={class:"jldocstring custom-block",open:""},f={class:"jldocstring custom-block",open:""},v={class:"jldocstring custom-block",open:""},C={class:"jldocstring custom-block",open:""};function E(x,e,F,B,j,z){const a=d("Badge");return r(),o("div",null,[e[33]||(e[33]=t("h1",{id:"Computer-Vision-Models-(Vision-API)",tabindex:"-1"},[i("Computer Vision Models ("),t("code",null,"Vision"),i(" API) "),t("a",{class:"header-anchor",href:"#Computer-Vision-Models-(Vision-API)","aria-label":'Permalink to "Computer Vision Models (`Vision` API) {#Computer-Vision-Models-(Vision-API)}"'},"​")],-1)),e[34]||(e[34]=t("h2",{id:"Native-Lux-Models",tabindex:"-1"},[i("Native Lux Models "),t("a",{class:"header-anchor",href:"#Native-Lux-Models","aria-label":'Permalink to "Native Lux Models {#Native-Lux-Models}"'},"​")],-1)),t("details",h,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Boltz.Vision.AlexNet",href:"#Boltz.Vision.AlexNet"},[t("span",{class:"jlbinding"},"Boltz.Vision.AlexNet")],-1)),e[1]||(e[1]=i()),l(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[2]||(e[2]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AlexNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create an AlexNet model (<a href="/Boltz.jl/previews/PR81/api/vision#krizhevsky2012imagenet">Krizhevsky <em>et al.</em>, 2012</a>).</p><p><strong>Keyword Arguments</strong></p><ul><li><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/alexnet.jl#L1-L10" target="_blank" rel="noreferrer">source</a></p>',5))]),t("details",c,[t("summary",null,[e[3]||(e[3]=t("a",{id:"Boltz.Vision.VGG",href:"#Boltz.Vision.VGG"},[t("span",{class:"jlbinding"},"Boltz.Vision.VGG")],-1)),e[4]||(e[4]=i()),l(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[5]||(e[5]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">VGG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imsize; config, inchannels, batchnorm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, nclasses, fcsize, dropout)</span></span></code></pre></div><p>Create a VGG model (<a href="/Boltz.jl/previews/PR81/api/vision#simonyan2014very">Simonyan, 2014</a>).</p><p><strong>Arguments</strong></p><ul><li><p><code>imsize</code>: input image width and height as a tuple</p></li><li><p><code>config</code>: the configuration for the convolution layers</p></li><li><p><code>inchannels</code>: number of input channels</p></li><li><p><code>batchnorm</code>: set to <code>true</code> to use batch normalization after each convolution</p></li><li><p><code>nclasses</code>: number of output classes</p></li><li><p><code>fcsize</code>: intermediate fully connected layer size</p></li><li><p><code>dropout</code>: dropout level between fully connected layers</p></li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/vgg.jl#L35-L49" target="_blank" rel="noreferrer">source</a></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">VGG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; batchnorm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a VGG model (<a href="/Boltz.jl/previews/PR81/api/vision#simonyan2014very">Simonyan, 2014</a>) with ImageNet Configuration.</p><p><strong>Arguments</strong></p><ul><li><code>depth::Int</code>: the depth of the VGG model. Choices: {<code>11</code>, <code>13</code>, <code>16</code>, <code>19</code>}.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>batchnorm = false</code>: set to <code>true</code> to use batch normalization after each convolution.</p></li><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</p></li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/vgg.jl#L65-L79" target="_blank" rel="noreferrer">source</a></p>',12))]),t("details",g,[t("summary",null,[e[6]||(e[6]=t("a",{id:"Boltz.Vision.VisionTransformer",href:"#Boltz.Vision.VisionTransformer"},[t("span",{class:"jlbinding"},"Boltz.Vision.VisionTransformer")],-1)),e[7]||(e[7]=i()),l(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[8]||(e[8]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">VisionTransformer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Creates a Vision Transformer model with the specified configuration.</p><p><strong>Arguments</strong></p><ul><li><code>name::Symbol</code>: name of the Vision Transformer model to create. The following models are available – <code>:tiny</code>, <code>:small</code>, <code>:base</code>, <code>:large</code>, <code>:huge</code>, <code>:giant</code>, <code>:gigantic</code>.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/vit.jl#L37-L51" target="_blank" rel="noreferrer">source</a></p>',7))]),e[35]||(e[35]=t("h2",{id:"Imported-from-Metalhead.jl",tabindex:"-1"},[i("Imported from Metalhead.jl "),t("a",{class:"header-anchor",href:"#Imported-from-Metalhead.jl","aria-label":'Permalink to "Imported from Metalhead.jl {#Imported-from-Metalhead.jl}"'},"​")],-1)),e[36]||(e[36]=t("div",{class:"tip custom-block"},[t("p",{class:"custom-block-title"},"Load Metalhead"),t("p",null,[i("You need to load "),t("code",null,"Metalhead"),i(" before using these models.")])],-1)),t("details",k,[t("summary",null,[e[9]||(e[9]=t("a",{id:"Boltz.Vision.ConvMixer",href:"#Boltz.Vision.ConvMixer"},[t("span",{class:"jlbinding"},"Boltz.Vision.ConvMixer")],-1)),e[10]||(e[10]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[11]||(e[11]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ConvMixer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a ConvMixer model (<a href="/Boltz.jl/previews/PR81/api/vision#trockman2022patches">Trockman and Kolter, 2022</a>).</p><p><strong>Arguments</strong></p><ul><li><code>name::Symbol</code>: The name of the ConvMixer model. Must be one of <code>:base</code>, <code>:small</code>, or <code>:large</code>.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/extensions.jl#L82-L96" target="_blank" rel="noreferrer">source</a></p>',7))]),t("details",u,[t("summary",null,[e[12]||(e[12]=t("a",{id:"Boltz.Vision.DenseNet",href:"#Boltz.Vision.DenseNet"},[t("span",{class:"jlbinding"},"Boltz.Vision.DenseNet")],-1)),e[13]||(e[13]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[14]||(e[14]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DenseNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a DenseNet model (<a href="/Boltz.jl/previews/PR81/api/vision#huang2017densely">Huang <em>et al.</em>, 2017</a>).</p><p><strong>Arguments</strong></p><ul><li><code>depth::Int</code>: The depth of the DenseNet model. Must be one of 121, 161, 169, or 201.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/extensions.jl#L48-L61" target="_blank" rel="noreferrer">source</a></p>',7))]),t("details",y,[t("summary",null,[e[15]||(e[15]=t("a",{id:"Boltz.Vision.GoogLeNet",href:"#Boltz.Vision.GoogLeNet"},[t("span",{class:"jlbinding"},"Boltz.Vision.GoogLeNet")],-1)),e[16]||(e[16]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[17]||(e[17]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">GoogLeNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a GoogLeNet model (<a href="/Boltz.jl/previews/PR81/api/vision#szegedy2015going">Szegedy <em>et al.</em>, 2015</a>).</p><p><strong>Keyword Arguments</strong></p><ul><li><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/extensions.jl#L36-L45" target="_blank" rel="noreferrer">source</a></p>',5))]),t("details",m,[t("summary",null,[e[18]||(e[18]=t("a",{id:"Boltz.Vision.MobileNet",href:"#Boltz.Vision.MobileNet"},[t("span",{class:"jlbinding"},"Boltz.Vision.MobileNet")],-1)),e[19]||(e[19]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[20]||(e[20]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">MobileNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a MobileNet model (<a href="/Boltz.jl/previews/PR81/api/vision#howard2017mobilenets">Howard, 2017</a>; <a href="/Boltz.jl/previews/PR81/api/vision#sandler2018mobilenetv2">Sandler <em>et al.</em>, 2018</a>; <a href="/Boltz.jl/previews/PR81/api/vision#howard2019searching">Howard <em>et al.</em>, 2019</a>).</p><p><strong>Arguments</strong></p><ul><li><code>name::Symbol</code>: The name of the MobileNet model. Must be one of <code>:v1</code>, <code>:v2</code>, <code>:v3_small</code>, or <code>:v3_large</code>.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/extensions.jl#L64-L79" target="_blank" rel="noreferrer">source</a></p>',7))]),t("details",b,[t("summary",null,[e[21]||(e[21]=t("a",{id:"Boltz.Vision.ResNet",href:"#Boltz.Vision.ResNet"},[t("span",{class:"jlbinding"},"Boltz.Vision.ResNet")],-1)),e[22]||(e[22]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[23]||(e[23]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ResNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a ResNet model (<a href="/Boltz.jl/previews/PR81/api/vision#he2016deep">He <em>et al.</em>, 2016</a>).</p><p><strong>Arguments</strong></p><ul><li><code>depth::Int</code>: The depth of the ResNet model. Must be one of 18, 34, 50, 101, or 152.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/extensions.jl#L1-L14" target="_blank" rel="noreferrer">source</a></p>',7))]),t("details",f,[t("summary",null,[e[24]||(e[24]=t("a",{id:"Boltz.Vision.ResNeXt",href:"#Boltz.Vision.ResNeXt"},[t("span",{class:"jlbinding"},"Boltz.Vision.ResNeXt")],-1)),e[25]||(e[25]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[26]||(e[26]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ResNeXt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; cardinality</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, base_width</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a ResNeXt model (<a href="/Boltz.jl/previews/PR81/api/vision#xie2017aggregated">Xie <em>et al.</em>, 2017</a>).</p><p><strong>Arguments</strong></p><ul><li><code>depth::Int</code>: The depth of the ResNeXt model. Must be one of 50, 101, or 152.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</p></li><li><p><code>cardinality</code>: The cardinality of the ResNeXt model. Defaults to 32.</p></li><li><p><code>base_width</code>: The base width of the ResNeXt model. Defaults to 8 for depth 101 and 4 otherwise.</p></li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/extensions.jl#L17-L33" target="_blank" rel="noreferrer">source</a></p>',7))]),t("details",v,[t("summary",null,[e[27]||(e[27]=t("a",{id:"Boltz.Vision.SqueezeNet",href:"#Boltz.Vision.SqueezeNet"},[t("span",{class:"jlbinding"},"Boltz.Vision.SqueezeNet")],-1)),e[28]||(e[28]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[29]||(e[29]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SqueezeNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a SqueezeNet model (<a href="/Boltz.jl/previews/PR81/api/vision#iandola2016squeezenetalexnetlevelaccuracy50x">Iandola <em>et al.</em>, 2016</a>).</p><p><strong>Keyword Arguments</strong></p><ul><li><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/extensions.jl#L99-L108" target="_blank" rel="noreferrer">source</a></p>',5))]),t("details",C,[t("summary",null,[e[30]||(e[30]=t("a",{id:"Boltz.Vision.WideResNet",href:"#Boltz.Vision.WideResNet"},[t("span",{class:"jlbinding"},"Boltz.Vision.WideResNet")],-1)),e[31]||(e[31]=i()),l(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[32]||(e[32]=s('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">WideResNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; pretrained</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a WideResNet model (<a href="/Boltz.jl/previews/PR81/api/vision#zagoruyko2017wideresidualnetworks">Zagoruyko and Komodakis, 2017</a>).</p><p><strong>Arguments</strong></p><ul><li><code>depth::Int</code>: The depth of the WideResNet model. Must be one of 18, 34, 50, 101, or 152.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><code>pretrained::Bool=false</code>: If <code>true</code>, loads pretrained weights when <code>LuxCore.setup</code> is called.</li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/8dc14a0658b1910bbe7df934fe0c4d2d3343e0cc/src/vision/extensions.jl#L111-L124" target="_blank" rel="noreferrer">source</a></p>',7))]),e[37]||(e[37]=s('<h2 id="Pretrained-Models" tabindex="-1">Pretrained Models <a class="header-anchor" href="#Pretrained-Models" aria-label="Permalink to &quot;Pretrained Models {#Pretrained-Models}&quot;">​</a></h2><div class="tip custom-block"><p class="custom-block-title">Load JLD2</p><p>You need to load <code>JLD2</code> before being able to load pretrained weights.</p></div><div class="tip custom-block"><p class="custom-block-title">Load Pretrained Weights</p><p>Pass <code>pretrained=true</code> to the model constructor to load the pretrained weights.</p></div><table tabindex="0"><thead><tr><th style="text-align:left;">MODEL</th><th style="text-align:center;">TOP 1 ACCURACY (%)</th><th style="text-align:center;">TOP 5 ACCURACY (%)</th></tr></thead><tbody><tr><td style="text-align:left;"><code>AlexNet()</code></td><td style="text-align:center;">54.48</td><td style="text-align:center;">77.72</td></tr><tr><td style="text-align:left;"><code>VGG(11)</code></td><td style="text-align:center;">67.35</td><td style="text-align:center;">87.91</td></tr><tr><td style="text-align:left;"><code>VGG(13)</code></td><td style="text-align:center;">68.40</td><td style="text-align:center;">88.48</td></tr><tr><td style="text-align:left;"><code>VGG(16)</code></td><td style="text-align:center;">70.24</td><td style="text-align:center;">89.80</td></tr><tr><td style="text-align:left;"><code>VGG(19)</code></td><td style="text-align:center;">71.09</td><td style="text-align:center;">90.27</td></tr><tr><td style="text-align:left;"><code>VGG(11; batchnorm=true)</code></td><td style="text-align:center;">69.09</td><td style="text-align:center;">88.94</td></tr><tr><td style="text-align:left;"><code>VGG(13; batchnorm=true)</code></td><td style="text-align:center;">69.66</td><td style="text-align:center;">89.49</td></tr><tr><td style="text-align:left;"><code>VGG(16; batchnorm=true)</code></td><td style="text-align:center;">72.11</td><td style="text-align:center;">91.02</td></tr><tr><td style="text-align:left;"><code>VGG(19; batchnorm=true)</code></td><td style="text-align:center;">72.95</td><td style="text-align:center;">91.32</td></tr><tr><td style="text-align:left;"><code>ResNet(18)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>ResNet(34)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>ResNet(50)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>ResNet(101)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>ResNet(152)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>ResNeXt(50; cardinality=32, base_width=4)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>ResNeXt(101; cardinality=32, base_width=8)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>ResNeXt(101; cardinality=64, base_width=4)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>SqueezeNet()</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>WideResNet(50)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr><tr><td style="text-align:left;"><code>WideResNet(101)</code></td><td style="text-align:center;">-</td><td style="text-align:center;">-</td></tr></tbody></table><div class="tip custom-block"><p class="custom-block-title">Pretrained Models from Metalhead</p><p>For Models imported from Metalhead, the pretrained weights can be loaded if they are available in Metalhead. Refer to the <a href="https://fluxml.ai/Metalhead.jl/stable/#Image-Classification" target="_blank" rel="noreferrer">Metalhead.jl docs</a> for a list of available pretrained models.</p></div><h3 id="preprocessing" tabindex="-1">Preprocessing <a class="header-anchor" href="#preprocessing" aria-label="Permalink to &quot;Preprocessing&quot;">​</a></h3><p>All the pretrained models require that the images be normalized with the parameters <code>mean = [0.485f0, 0.456f0, 0.406f0]</code> and <code>std = [0.229f0, 0.224f0, 0.225f0]</code>.</p><hr><h1 id="bibliography" tabindex="-1">Bibliography <a class="header-anchor" href="#bibliography" aria-label="Permalink to &quot;Bibliography&quot;">​</a></h1><ul><li><p>He, K.; Zhang, X.; Ren, S. and Sun, J. (2016). <em>Deep residual learning for image recognition</em>. In: <em>Proceedings of the IEEE conference on computer vision and pattern recognition</em>; pp. 770–778.</p></li><li><p>Howard, A.; Sandler, M.; Chu, G.; Chen, L.-C.; Chen, B.; Tan, M.; Wang, W.; Zhu, Y.; Pang, R.; Vasudevan, V. and others (2019). <em>Searching for mobilenetv3</em>. In: <em>Proceedings of the IEEE/CVF international conference on computer vision</em>; pp. 1314–1324.</p></li><li><p>Howard, A. G. (2017). <em>MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications</em>, arXiv preprint arXiv:1704.04861.</p></li><li><p>Huang, G.; Liu, Z.; Van Der Maaten, L. and Weinberger, K. Q. (2017). <em>Densely connected convolutional networks</em>. In: <em>Proceedings of the IEEE conference on computer vision and pattern recognition</em>; pp. 4700–4708.</p></li><li><p>Iandola, F. N.; Han, S.; Moskewicz, M. W.; Ashraf, K.; Dally, W. J. and Keutzer, K. (2016). <a href="https://arxiv.org/abs/1602.07360" target="_blank" rel="noreferrer"><em>SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and &lt;0.5MB model size</em></a>, <a href="https://arxiv.org/abs/1602.07360" target="_blank" rel="noreferrer">arXiv:1602.07360 [cs.CV]</a>.</p></li><li><p>Krizhevsky, A.; Sutskever, I. and Hinton, G. E. (2012). <em>Imagenet classification with deep convolutional neural networks</em>. Advances in neural information processing systems <strong>25</strong>.</p></li><li><p>Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A. and Chen, L.-C. (2018). <em>Mobilenetv2: Inverted residuals and linear bottlenecks</em>. In: <em>Proceedings of the IEEE conference on computer vision and pattern recognition</em>; pp. 4510–4520.</p></li><li><p>Simonyan, K. (2014). <em>Very deep convolutional networks for large-scale image recognition</em>, arXiv preprint arXiv:1409.1556.</p></li><li><p>Szegedy, C.; Liu, W.; Jia, Y.; Sermanet, P.; Reed, S.; Anguelov, D.; Erhan, D.; Vanhoucke, V. and Rabinovich, A. (2015). <em>Going deeper with convolutions</em>. In: <em>Proceedings of the IEEE conference on computer vision and pattern recognition</em>; pp. 1–9.</p></li><li><p>Trockman, A. and Kolter, J. Z. (2022). <em>Patches are all you need?</em> arXiv preprint arXiv:2201.09792.</p></li><li><p>Xie, S.; Girshick, R.; Dollár, P.; Tu, Z. and He, K. (2017). <em>Aggregated residual transformations for deep neural networks</em>. In: <em>Proceedings of the IEEE conference on computer vision and pattern recognition</em>; pp. 1492–1500.</p></li><li><p>Zagoruyko, S. and Komodakis, N. (2017). <a href="https://arxiv.org/abs/1605.07146" target="_blank" rel="noreferrer"><em>Wide Residual Networks</em></a>, <a href="https://arxiv.org/abs/1605.07146" target="_blank" rel="noreferrer">arXiv:1605.07146 [cs.CV]</a>.</p></li></ul>',10))])}const V=n(p,[["render",E]]);export{A as __pageData,V as default};
