import{_ as o,C as r,c as p,o as d,j as t,a,a2 as l,G as i,w as n}from"./chunks/framework.CKUMLBlJ.js";const A=JSON.parse('{"title":"Private API","description":"","frontmatter":{},"headers":[],"relativePath":"api/private.md","filePath":"api/private.md","lastUpdated":null}'),h={name:"api/private.md"},u={class:"jldocstring custom-block",open:""},k={class:"jldocstring custom-block",open:""},c={class:"jldocstring custom-block",open:""},b={class:"jldocstring custom-block",open:""};function f(_,s,g,y,T,m){const e=r("Badge");return d(),p("div",null,[s[16]||(s[16]=t("h1",{id:"Private-API",tabindex:"-1"},[a("Private API "),t("a",{class:"header-anchor",href:"#Private-API","aria-label":'Permalink to "Private API {#Private-API}"'},"​")],-1)),s[17]||(s[17]=t("p",null,"This is the private API reference for Boltz.jl. You know what this means. Don't use these functions!",-1)),t("details",u,[t("summary",null,[s[0]||(s[0]=t("a",{id:"Boltz.Utils.fast_chunk-Tuple{Int64, Int64}",href:"#Boltz.Utils.fast_chunk-Tuple{Int64, Int64}"},[t("span",{class:"jlbinding"},"Boltz.Utils.fast_chunk")],-1)),s[1]||(s[1]=a()),i(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[3]||(s[3]=l('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">fast_chunk</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val{n}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val{dim}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Type-stable and faster version of <code>MLUtils.chunk</code>.</p>',2)),i(e,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[2]||(s[2]=[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/6cce8c93545b028621f6fdddb61e648b5e92233f/src/utils.jl#L11-L15",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",k,[t("summary",null,[s[4]||(s[4]=t("a",{id:"Boltz.Utils.flatten_spatial-Union{Tuple{AbstractArray{T, 4}}, Tuple{T}} where T",href:"#Boltz.Utils.flatten_spatial-Union{Tuple{AbstractArray{T, 4}}, Tuple{T}} where T"},[t("span",{class:"jlbinding"},"Boltz.Utils.flatten_spatial")],-1)),s[5]||(s[5]=a()),i(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[7]||(s[7]=l('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">flatten_spatial</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 4}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Flattens the first 2 dimensions of <code>x</code>, and permutes the remaining dimensions to (2, 1, 3).</p>',2)),i(e,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[6]||(s[6]=[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/6cce8c93545b028621f6fdddb61e648b5e92233f/src/utils.jl#L27-L31",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",c,[t("summary",null,[s[8]||(s[8]=t("a",{id:"Boltz.Utils.second_dim_mean-Tuple{Any}",href:"#Boltz.Utils.second_dim_mean-Tuple{Any}"},[t("span",{class:"jlbinding"},"Boltz.Utils.second_dim_mean")],-1)),s[9]||(s[9]=a()),i(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[11]||(s[11]=l('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">second_dim_mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span></code></pre></div><p>Computes the mean of <code>x</code> along dimension <code>2</code>.</p>',2)),i(e,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[10]||(s[10]=[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/6cce8c93545b028621f6fdddb61e648b5e92233f/src/utils.jl#L37-L41",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",b,[t("summary",null,[s[12]||(s[12]=t("a",{id:"Boltz.Utils.should_type_assert-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T",href:"#Boltz.Utils.should_type_assert-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T"},[t("span",{class:"jlbinding"},"Boltz.Utils.should_type_assert")],-1)),s[13]||(s[13]=a()),i(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[15]||(s[15]=l('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">should_type_assert</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span></code></pre></div><p>In certain cases, to ensure type-stability we want to add type-asserts. But this won&#39;t work for exotic types like <code>ForwardDiff.Dual</code>. We use this function to check if we should add a type-assert for <code>x</code>.</p>',2)),i(e,{type:"info",class:"source-link",text:"source"},{default:n(()=>s[14]||(s[14]=[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/6cce8c93545b028621f6fdddb61e648b5e92233f/src/utils.jl#L44-L50",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const E=o(h,[["render",f]]);export{A as __pageData,E as default};
