import{_ as s,c as Q,j as t,a,a4 as T,o as e}from"./chunks/framework.2hqcchSi.js";const V2=JSON.parse('{"title":"Boltz.Basis API Reference","description":"","frontmatter":{},"headers":[],"relativePath":"api/basis.md","filePath":"api/basis.md","lastUpdated":null}'),l={name:"api/basis.md"},o=t("h1",{id:"Boltz.Basis-API-Reference",tabindex:"-1"},[t("code",null,"Boltz.Basis"),a(" API Reference "),t("a",{class:"header-anchor",href:"#Boltz.Basis-API-Reference","aria-label":'Permalink to "`Boltz.Basis` API Reference {#Boltz.Basis-API-Reference}"'},"​")],-1),n=t("div",{class:"warning custom-block"},[t("p",{class:"custom-block-title"},"Warning"),t("p",null,"The function calls for these basis functions should be considered experimental and are subject to change without deprecation. However, the functions themselves are stable and can be freely used in combination with the other Layers and Models.")],-1),d={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},r=t("a",{id:"Boltz.Basis.Chebyshev-Tuple{Any}",href:"#Boltz.Basis.Chebyshev-Tuple{Any}"},"#",-1),i=t("b",null,[t("u",null,"Boltz.Basis.Chebyshev")],-1),m=t("i",null,"Method",-1),h=T("",1),p={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},c={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"25.599ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 11314.7 1000","aria-hidden":"true"},g=T("",1),H=[g],u=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mo",{stretchy:"false"},"["),t("msub",null,[t("mi",null,"T"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mn",null,"0")])]),t("mo",{stretchy:"false"},"("),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",null,","),t("msub",null,[t("mi",null,"T"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mn",null,"1")])]),t("mo",{stretchy:"false"},"("),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",null,","),t("mo",null,"…"),t("mo",null,","),t("msub",null,[t("mi",null,"T"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",null,"n"),t("mo",null,"−"),t("mn",null,"1")])]),t("mo",{stretchy:"false"},"("),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",{stretchy:"false"},"]")])],-1),_={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},k={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.666ex"},xmlns:"http://www.w3.org/2000/svg",width:"4.934ex",height:"2.363ex",role:"img",focusable:"false",viewBox:"0 -750 2181 1044.2","aria-hidden":"true"},w=T("",1),f=[w],y=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msub",null,[t("mi",null,"T"),t("mi",null,"j")]),t("mo",{stretchy:"false"},"("),t("mo",null,"."),t("mo",{stretchy:"false"},")")])],-1),x={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},V={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.462ex"},xmlns:"http://www.w3.org/2000/svg",width:"2.619ex",height:"2.393ex",role:"img",focusable:"false",viewBox:"0 -853.7 1157.6 1057.7","aria-hidden":"true"},L=T("",1),M=[L],b=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msup",null,[t("mi",null,"j"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",null,"t"),t("mi",null,"h")])])])],-1),Z=t("p",null,[t("strong",null,"Arguments")],-1),v=t("ul",null,[t("li",null,[t("code",null,"n"),a(": number of terms in the polynomial expansion.")])],-1),C=t("p",null,[t("strong",null,"Keyword Arguments")],-1),A=t("ul",null,[t("li",null,[t("code",null,"dim::Int=1"),a(": The dimension along which the basis functions are applied.")])],-1),D=t("p",null,[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/930c0e37da7f51a4ffbde6cf63d8d4b6fa4ee25d/src/basis.jl#L48",target:"_blank",rel:"noreferrer"},"source")],-1),j=t("br",null,null,-1),B={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},E=t("a",{id:"Boltz.Basis.Cos-Tuple{Any}",href:"#Boltz.Basis.Cos-Tuple{Any}"},"#",-1),S=t("b",null,[t("u",null,"Boltz.Basis.Cos")],-1),F=t("i",null,"Method",-1),P=T("",1),I={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},R={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"28.038ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 12392.7 1000","aria-hidden":"true"},N=T("",1),z=[N],O=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mo",{stretchy:"false"},"["),t("mi",null,"cos"),t("mo",{"data-mjx-texclass":"NONE"},"⁡"),t("mo",{stretchy:"false"},"("),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",null,","),t("mi",null,"cos"),t("mo",{"data-mjx-texclass":"NONE"},"⁡"),t("mo",{stretchy:"false"},"("),t("mn",null,"2"),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",null,","),t("mo",null,"…"),t("mo",null,","),t("mi",null,"cos"),t("mo",{"data-mjx-texclass":"NONE"},"⁡"),t("mo",{stretchy:"false"},"("),t("mi",null,"n"),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",{stretchy:"false"},"]")])],-1),J=t("p",null,[t("strong",null,"Arguments")],-1),G=t("ul",null,[t("li",null,[t("code",null,"n"),a(": number of terms in the cosine expansion.")])],-1),X=t("p",null,[t("strong",null,"Keyword Arguments")],-1),K=t("ul",null,[t("li",null,[t("code",null,"dim::Int=1"),a(": The dimension along which the basis functions are applied.")])],-1),$=t("p",null,[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/930c0e37da7f51a4ffbde6cf63d8d4b6fa4ee25d/src/basis.jl#L81",target:"_blank",rel:"noreferrer"},"source")],-1),U=t("br",null,null,-1),W={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},q=t("a",{id:"Boltz.Basis.Fourier-Tuple{Any}",href:"#Boltz.Basis.Fourier-Tuple{Any}"},"#",-1),Y=t("b",null,[t("u",null,"Boltz.Basis.Fourier")],-1),t1=t("i",null,"Method",-1),a1=T("",2),T1={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},Q1={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-3.731ex"},xmlns:"http://www.w3.org/2000/svg",width:"31.946ex",height:"8.593ex",role:"img",focusable:"false",viewBox:"0 -2149 14120.1 3798","aria-hidden":"true"},e1=T("",1),s1=[e1],l1=t("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[t("msub",null,[t("mi",null,"F"),t("mi",null,"j")]),t("mo",{stretchy:"false"},"("),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",null,"="),t("mrow",{"data-mjx-texclass":"INNER"},[t("mo",{"data-mjx-texclass":"OPEN"},"{"),t("mtable",{columnalign:"left left",columnspacing:"1em",rowspacing:".2em"},[t("mtr",null,[t("mtd",null,[t("mi",null,"c"),t("mi",null,"o"),t("mi",null,"s"),t("mrow",{"data-mjx-texclass":"INNER"},[t("mo",{"data-mjx-texclass":"OPEN"},"("),t("mfrac",null,[t("mi",null,"j"),t("mn",null,"2")]),t("mi",null,"x"),t("mo",{"data-mjx-texclass":"CLOSE"},")")])]),t("mtd",null,[t("mtext",null,"if "),t("mi",null,"j"),t("mtext",null," is even")])]),t("mtr",null,[t("mtd",null,[t("mi",null,"s"),t("mi",null,"i"),t("mi",null,"n"),t("mrow",{"data-mjx-texclass":"INNER"},[t("mo",{"data-mjx-texclass":"OPEN"},"("),t("mfrac",null,[t("mi",null,"j"),t("mn",null,"2")]),t("mi",null,"x"),t("mo",{"data-mjx-texclass":"CLOSE"},")")])]),t("mtd",null,[t("mtext",null,"if "),t("mi",null,"j"),t("mtext",null," is odd")])])]),t("mo",{"data-mjx-texclass":"CLOSE",fence:"true",stretchy:"true",symmetric:"true"})])])],-1),o1=t("p",null,[t("strong",null,"Arguments")],-1),n1=t("ul",null,[t("li",null,[t("code",null,"n"),a(": number of terms in the Fourier expansion.")])],-1),d1=t("p",null,[t("strong",null,"Keyword Arguments")],-1),r1=t("ul",null,[t("li",null,[t("code",null,"dim::Int=1"),a(": The dimension along which the basis functions are applied.")])],-1),i1=t("p",null,[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/930c0e37da7f51a4ffbde6cf63d8d4b6fa4ee25d/src/basis.jl#L96",target:"_blank",rel:"noreferrer"},"source")],-1),m1=t("br",null,null,-1),h1={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},p1=t("a",{id:"Boltz.Basis.Legendre-Tuple{Any}",href:"#Boltz.Basis.Legendre-Tuple{Any}"},"#",-1),c1=t("b",null,[t("u",null,"Boltz.Basis.Legendre")],-1),g1=t("i",null,"Method",-1),H1=T("",1),u1={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},_1={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"25.993ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 11488.7 1000","aria-hidden":"true"},k1=T("",1),w1=[k1],f1=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mo",{stretchy:"false"},"["),t("msub",null,[t("mi",null,"P"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mn",null,"0")])]),t("mo",{stretchy:"false"},"("),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",null,","),t("msub",null,[t("mi",null,"P"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mn",null,"1")])]),t("mo",{stretchy:"false"},"("),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",null,","),t("mo",null,"…"),t("mo",null,","),t("msub",null,[t("mi",null,"P"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",null,"n"),t("mo",null,"−"),t("mn",null,"1")])]),t("mo",{stretchy:"false"},"("),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",{stretchy:"false"},"]")])],-1),y1={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},x1={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.666ex"},xmlns:"http://www.w3.org/2000/svg",width:"5.066ex",height:"2.363ex",role:"img",focusable:"false",viewBox:"0 -750 2239 1044.2","aria-hidden":"true"},V1=T("",1),L1=[V1],M1=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msub",null,[t("mi",null,"P"),t("mi",null,"j")]),t("mo",{stretchy:"false"},"("),t("mo",null,"."),t("mo",{stretchy:"false"},")")])],-1),b1={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Z1={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.462ex"},xmlns:"http://www.w3.org/2000/svg",width:"2.619ex",height:"2.393ex",role:"img",focusable:"false",viewBox:"0 -853.7 1157.6 1057.7","aria-hidden":"true"},v1=T("",1),C1=[v1],A1=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msup",null,[t("mi",null,"j"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",null,"t"),t("mi",null,"h")])])])],-1),D1=t("p",null,[t("strong",null,"Arguments")],-1),j1=t("ul",null,[t("li",null,[t("code",null,"n"),a(": number of terms in the polynomial expansion.")])],-1),B1=t("p",null,[t("strong",null,"Keyword Arguments")],-1),E1=t("ul",null,[t("li",null,[t("code",null,"dim::Int=1"),a(": The dimension along which the basis functions are applied.")])],-1),S1=t("p",null,[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/930c0e37da7f51a4ffbde6cf63d8d4b6fa4ee25d/src/basis.jl#L137",target:"_blank",rel:"noreferrer"},"source")],-1),F1=t("br",null,null,-1),P1={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},I1=t("a",{id:"Boltz.Basis.Polynomial-Tuple{Any}",href:"#Boltz.Basis.Polynomial-Tuple{Any}"},"#",-1),R1=t("b",null,[t("u",null,"Boltz.Basis.Polynomial")],-1),N1=t("i",null,"Method",-1),z1=T("",1),O1={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},J1={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"15.461ex",height:"2.587ex",role:"img",focusable:"false",viewBox:"0 -893.3 6833.7 1143.3","aria-hidden":"true"},G1=T("",1),X1=[G1],K1=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mo",{stretchy:"false"},"["),t("mn",null,"1"),t("mo",null,","),t("mi",null,"x"),t("mo",null,","),t("mo",null,"…"),t("mo",null,","),t("msup",null,[t("mi",null,"x"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mo",{stretchy:"false"},"("),t("mi",null,"n"),t("mo",null,"−"),t("mn",null,"1"),t("mo",{stretchy:"false"},")")])]),t("mo",{stretchy:"false"},"]")])],-1),$1=t("p",null,[t("strong",null,"Arguments")],-1),U1=t("ul",null,[t("li",null,[t("code",null,"n"),a(": number of terms in the polynomial expansion.")])],-1),W1=t("p",null,[t("strong",null,"Keyword Arguments")],-1),q1=t("ul",null,[t("li",null,[t("code",null,"dim::Int=1"),a(": The dimension along which the basis functions are applied.")])],-1),Y1=t("p",null,[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/930c0e37da7f51a4ffbde6cf63d8d4b6fa4ee25d/src/basis.jl#L169",target:"_blank",rel:"noreferrer"},"source")],-1),t2=t("br",null,null,-1),a2={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},T2=t("a",{id:"Boltz.Basis.Sin-Tuple{Any}",href:"#Boltz.Basis.Sin-Tuple{Any}"},"#",-1),Q2=t("b",null,[t("u",null,"Boltz.Basis.Sin")],-1),e2=t("i",null,"Method",-1),s2=T("",1),l2={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},o2={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"27.291ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 12062.7 1000","aria-hidden":"true"},n2=T("",1),d2=[n2],r2=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mo",{stretchy:"false"},"["),t("mi",null,"sin"),t("mo",{"data-mjx-texclass":"NONE"},"⁡"),t("mo",{stretchy:"false"},"("),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",null,","),t("mi",null,"sin"),t("mo",{"data-mjx-texclass":"NONE"},"⁡"),t("mo",{stretchy:"false"},"("),t("mn",null,"2"),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",null,","),t("mo",null,"…"),t("mo",null,","),t("mi",null,"sin"),t("mo",{"data-mjx-texclass":"NONE"},"⁡"),t("mo",{stretchy:"false"},"("),t("mi",null,"n"),t("mi",null,"x"),t("mo",{stretchy:"false"},")"),t("mo",{stretchy:"false"},"]")])],-1),i2=t("p",null,[t("strong",null,"Arguments")],-1),m2=t("ul",null,[t("li",null,[t("code",null,"n"),a(": number of terms in the sine expansion.")])],-1),h2=t("p",null,[t("strong",null,"Keyword Arguments")],-1),p2=t("ul",null,[t("li",null,[t("code",null,"dim::Int=1"),a(": The dimension along which the basis functions are applied.")])],-1),c2=t("p",null,[t("a",{href:"https://github.com/LuxDL/Boltz.jl/blob/930c0e37da7f51a4ffbde6cf63d8d4b6fa4ee25d/src/basis.jl#L66",target:"_blank",rel:"noreferrer"},"source")],-1),g2=t("br",null,null,-1);function H2(u2,_2,k2,w2,f2,y2){return e(),Q("div",null,[o,n,t("div",d,[r,a(" "),i,a(" — "),m,a(". "),h,t("p",null,[a("Constructs a Chebyshev basis of the form "),t("mjx-container",p,[(e(),Q("svg",c,H)),u]),a(" where "),t("mjx-container",_,[(e(),Q("svg",k,f)),y]),a(" is the "),t("mjx-container",x,[(e(),Q("svg",V,M)),b]),a(" Chebyshev polynomial of the first kind.")]),Z,v,C,A,D]),j,t("div",B,[E,a(" "),S,a(" — "),F,a(". "),P,t("p",null,[a("Constructs a cosine basis of the form "),t("mjx-container",I,[(e(),Q("svg",R,z)),O]),a(".")]),J,G,X,K,$]),U,t("div",W,[q,a(" "),Y,a(" — "),t1,a(". "),a1,t("mjx-container",T1,[(e(),Q("svg",Q1,s1)),l1]),o1,n1,d1,r1,i1]),m1,t("div",h1,[p1,a(" "),c1,a(" — "),g1,a(". "),H1,t("p",null,[a("Constructs a Legendre basis of the form "),t("mjx-container",u1,[(e(),Q("svg",_1,w1)),f1]),a(" where "),t("mjx-container",y1,[(e(),Q("svg",x1,L1)),M1]),a(" is the "),t("mjx-container",b1,[(e(),Q("svg",Z1,C1)),A1]),a(" Legendre polynomial.")]),D1,j1,B1,E1,S1]),F1,t("div",P1,[I1,a(" "),R1,a(" — "),N1,a(". "),z1,t("p",null,[a("Constructs a Polynomial basis of the form "),t("mjx-container",O1,[(e(),Q("svg",J1,X1)),K1]),a(".")]),$1,U1,W1,q1,Y1]),t2,t("div",a2,[T2,a(" "),Q2,a(" — "),e2,a(". "),s2,t("p",null,[a("Constructs a sine basis of the form "),t("mjx-container",l2,[(e(),Q("svg",o2,d2)),r2]),a(".")]),i2,m2,h2,p2,c2]),g2])}const L2=s(l,[["render",H2]]);export{V2 as __pageData,L2 as default};
