var Ri=Object.defineProperty;var Zi=(t,e,i)=>e in t?Ri(t,e,{enumerable:!0,configurable:!0,writable:!0,value:i}):t[e]=i;var p=(t,e,i)=>(Zi(t,typeof e!="symbol"?e+"":e,i),i);import{z as W,r as me,A as li,o as hi,B as $e,d as Ji,C as Wi,D as Ki,c as Yi,E as Xi,F as Ut,G as Gi,e as Qi}from"./entry.2f09776e.js";var Nt=window,ke=Nt.ShadowRoot&&(Nt.ShadyCSS===void 0||Nt.ShadyCSS.nativeShadow)&&"adoptedStyleSheets"in Document.prototype&&"replace"in CSSStyleSheet.prototype,xe=Symbol(),He=new WeakMap,di=class{constructor(e,i,o){if(this._$cssResult$=!0,o!==xe)throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");this.cssText=e,this.t=i}get styleSheet(){let e=this.o,i=this.t;if(ke&&e===void 0){let o=i!==void 0&&i.length===1;o&&(e=He.get(i)),e===void 0&&((this.o=e=new CSSStyleSheet).replaceSync(this.cssText),o&&He.set(i,e))}return e}toString(){return this.cssText}},to=t=>new di(typeof t=="string"?t:t+"",void 0,xe),eo=(t,...e)=>{let i=t.length===1?t[0]:e.reduce((o,n,s)=>o+(a=>{if(a._$cssResult$===!0)return a.cssText;if(typeof a=="number")return a;throw Error("Value passed to 'css' function must be a 'css' function result: "+a+". Use 'unsafeCSS' to pass non-literal values, but take care to ensure page security.")})(n)+t[s+1],t[0]);return new di(i,t,xe)},io=(t,e)=>{ke?t.adoptedStyleSheets=e.map(i=>i instanceof CSSStyleSheet?i:i.styleSheet):e.forEach(i=>{let o=document.createElement("style"),n=Nt.litNonce;n!==void 0&&o.setAttribute("nonce",n),o.textContent=i.cssText,t.appendChild(o)})},qe=ke?t=>t:t=>t instanceof CSSStyleSheet?(e=>{let i="";for(let o of e.cssRules)i+=o.cssText;return to(i)})(t):t,ie,Vt=window,Re=Vt.trustedTypes,oo=Re?Re.emptyScript:"",oe=Vt.reactiveElementPolyfillSupport,ve={toAttribute(t,e){switch(e){case Boolean:t=t?oo:null;break;case Object:case Array:t=t==null?t:JSON.stringify(t)}return t},fromAttribute(t,e){let i=t;switch(e){case Boolean:i=t!==null;break;case Number:i=t===null?null:Number(t);break;case Object:case Array:try{i=JSON.parse(t)}catch{i=null}}return i}},ui=(t,e)=>e!==t&&(e==e||t==t),ne={attribute:!0,type:String,converter:ve,reflect:!1,hasChanged:ui},_e="finalized",mt=class extends HTMLElement{constructor(){super(),this._$Ei=new Map,this.isUpdatePending=!1,this.hasUpdated=!1,this._$El=null,this._$Eu()}static addInitializer(t){var e;this.finalize(),((e=this.h)!==null&&e!==void 0?e:this.h=[]).push(t)}static get observedAttributes(){this.finalize();let t=[];return this.elementProperties.forEach((e,i)=>{let o=this._$Ep(i,e);o!==void 0&&(this._$Ev.set(o,i),t.push(o))}),t}static createProperty(t,e=ne){if(e.state&&(e.attribute=!1),this.finalize(),this.elementProperties.set(t,e),!e.noAccessor&&!this.prototype.hasOwnProperty(t)){let i=typeof t=="symbol"?Symbol():"__"+t,o=this.getPropertyDescriptor(t,i,e);o!==void 0&&Object.defineProperty(this.prototype,t,o)}}static getPropertyDescriptor(t,e,i){return{get(){return this[e]},set(o){let n=this[t];this[e]=o,this.requestUpdate(t,n,i)},configurable:!0,enumerable:!0}}static getPropertyOptions(t){return this.elementProperties.get(t)||ne}static finalize(){if(this.hasOwnProperty(_e))return!1;this[_e]=!0;let t=Object.getPrototypeOf(this);if(t.finalize(),t.h!==void 0&&(this.h=[...t.h]),this.elementProperties=new Map(t.elementProperties),this._$Ev=new Map,this.hasOwnProperty("properties")){let e=this.properties,i=[...Object.getOwnPropertyNames(e),...Object.getOwnPropertySymbols(e)];for(let o of i)this.createProperty(o,e[o])}return this.elementStyles=this.finalizeStyles(this.styles),!0}static finalizeStyles(t){let e=[];if(Array.isArray(t)){let i=new Set(t.flat(1/0).reverse());for(let o of i)e.unshift(qe(o))}else t!==void 0&&e.push(qe(t));return e}static _$Ep(t,e){let i=e.attribute;return i===!1?void 0:typeof i=="string"?i:typeof t=="string"?t.toLowerCase():void 0}_$Eu(){var t;this._$E_=new Promise(e=>this.enableUpdating=e),this._$AL=new Map,this._$Eg(),this.requestUpdate(),(t=this.constructor.h)===null||t===void 0||t.forEach(e=>e(this))}addController(t){var e,i;((e=this._$ES)!==null&&e!==void 0?e:this._$ES=[]).push(t),this.renderRoot!==void 0&&this.isConnected&&((i=t.hostConnected)===null||i===void 0||i.call(t))}removeController(t){var e;(e=this._$ES)===null||e===void 0||e.splice(this._$ES.indexOf(t)>>>0,1)}_$Eg(){this.constructor.elementProperties.forEach((t,e)=>{this.hasOwnProperty(e)&&(this._$Ei.set(e,this[e]),delete this[e])})}createRenderRoot(){var t;let e=(t=this.shadowRoot)!==null&&t!==void 0?t:this.attachShadow(this.constructor.shadowRootOptions);return io(e,this.constructor.elementStyles),e}connectedCallback(){var t;this.renderRoot===void 0&&(this.renderRoot=this.createRenderRoot()),this.enableUpdating(!0),(t=this._$ES)===null||t===void 0||t.forEach(e=>{var i;return(i=e.hostConnected)===null||i===void 0?void 0:i.call(e)})}enableUpdating(t){}disconnectedCallback(){var t;(t=this._$ES)===null||t===void 0||t.forEach(e=>{var i;return(i=e.hostDisconnected)===null||i===void 0?void 0:i.call(e)})}attributeChangedCallback(t,e,i){this._$AK(t,i)}_$EO(t,e,i=ne){var o;let n=this.constructor._$Ep(t,i);if(n!==void 0&&i.reflect===!0){let s=(((o=i.converter)===null||o===void 0?void 0:o.toAttribute)!==void 0?i.converter:ve).toAttribute(e,i.type);this._$El=t,s==null?this.removeAttribute(n):this.setAttribute(n,s),this._$El=null}}_$AK(t,e){var i;let o=this.constructor,n=o._$Ev.get(t);if(n!==void 0&&this._$El!==n){let s=o.getPropertyOptions(n),a=typeof s.converter=="function"?{fromAttribute:s.converter}:((i=s.converter)===null||i===void 0?void 0:i.fromAttribute)!==void 0?s.converter:ve;this._$El=n,this[n]=a.fromAttribute(e,s.type),this._$El=null}}requestUpdate(t,e,i){let o=!0;t!==void 0&&(((i=i||this.constructor.getPropertyOptions(t)).hasChanged||ui)(this[t],e)?(this._$AL.has(t)||this._$AL.set(t,e),i.reflect===!0&&this._$El!==t&&(this._$EC===void 0&&(this._$EC=new Map),this._$EC.set(t,i))):o=!1),!this.isUpdatePending&&o&&(this._$E_=this._$Ej())}async _$Ej(){this.isUpdatePending=!0;try{await this._$E_}catch(e){Promise.reject(e)}let t=this.scheduleUpdate();return t!=null&&await t,!this.isUpdatePending}scheduleUpdate(){return this.performUpdate()}performUpdate(){var t;if(!this.isUpdatePending)return;this.hasUpdated,this._$Ei&&(this._$Ei.forEach((o,n)=>this[n]=o),this._$Ei=void 0);let e=!1,i=this._$AL;try{e=this.shouldUpdate(i),e?(this.willUpdate(i),(t=this._$ES)===null||t===void 0||t.forEach(o=>{var n;return(n=o.hostUpdate)===null||n===void 0?void 0:n.call(o)}),this.update(i)):this._$Ek()}catch(o){throw e=!1,this._$Ek(),o}e&&this._$AE(i)}willUpdate(t){}_$AE(t){var e;(e=this._$ES)===null||e===void 0||e.forEach(i=>{var o;return(o=i.hostUpdated)===null||o===void 0?void 0:o.call(i)}),this.hasUpdated||(this.hasUpdated=!0,this.firstUpdated(t)),this.updated(t)}_$Ek(){this._$AL=new Map,this.isUpdatePending=!1}get updateComplete(){return this.getUpdateComplete()}getUpdateComplete(){return this._$E_}shouldUpdate(t){return!0}update(t){this._$EC!==void 0&&(this._$EC.forEach((e,i)=>this._$EO(i,this[i],e)),this._$EC=void 0),this._$Ek()}updated(t){}firstUpdated(t){}};mt[_e]=!0,mt.elementProperties=new Map,mt.elementStyles=[],mt.shadowRootOptions={mode:"open"},oe==null||oe({ReactiveElement:mt}),((ie=Vt.reactiveElementVersions)!==null&&ie!==void 0?ie:Vt.reactiveElementVersions=[]).push("1.6.3");var re,Ft=window,gt=Ft.trustedTypes,Ze=gt?gt.createPolicy("lit-html",{createHTML:t=>t}):void 0,ye="$lit$",nt=`lit$${(Math.random()+"").slice(9)}$`,ci="?"+nt,no=`<${ci}>`,dt=document,xt=()=>dt.createComment(""),Et=t=>t===null||typeof t!="object"&&typeof t!="function",pi=Array.isArray,ro=t=>pi(t)||typeof(t==null?void 0:t[Symbol.iterator])=="function",se=`[ 	
\f\r]`,St=/<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g,Je=/-->/g,We=/>/g,at=RegExp(`>|${se}(?:([^\\s"'>=/]+)(${se}*=${se}*(?:[^ 	
\f\r"'\`<>=]|("|')|))|$)`,"g"),Ke=/'/g,Ye=/"/g,fi=/^(?:script|style|textarea|title)$/i,so=t=>(e,...i)=>({_$litType$:t,strings:e,values:i}),w=so(1),bt=Symbol.for("lit-noChange"),U=Symbol.for("lit-nothing"),Xe=new WeakMap,lt=dt.createTreeWalker(dt,129,null,!1);function mi(t,e){if(!Array.isArray(t)||!t.hasOwnProperty("raw"))throw Error("invalid template strings array");return Ze!==void 0?Ze.createHTML(e):e}var ao=(t,e)=>{let i=t.length-1,o=[],n,s=e===2?"<svg>":"",a=St;for(let h=0;h<i;h++){let r=t[h],l,d,u=-1,c=0;for(;c<r.length&&(a.lastIndex=c,d=a.exec(r),d!==null);)c=a.lastIndex,a===St?d[1]==="!--"?a=Je:d[1]!==void 0?a=We:d[2]!==void 0?(fi.test(d[2])&&(n=RegExp("</"+d[2],"g")),a=at):d[3]!==void 0&&(a=at):a===at?d[0]===">"?(a=n??St,u=-1):d[1]===void 0?u=-2:(u=a.lastIndex-d[2].length,l=d[1],a=d[3]===void 0?at:d[3]==='"'?Ye:Ke):a===Ye||a===Ke?a=at:a===Je||a===We?a=St:(a=at,n=void 0);let f=a===at&&t[h+1].startsWith("/>")?" ":"";s+=a===St?r+no:u>=0?(o.push(l),r.slice(0,u)+ye+r.slice(u)+nt+f):r+nt+(u===-2?(o.push(void 0),h):f)}return[mi(t,s+(t[i]||"<?>")+(e===2?"</svg>":"")),o]},Bt=class{constructor({strings:t,_$litType$:e},i){let o;this.parts=[];let n=0,s=0,a=t.length-1,h=this.parts,[r,l]=ao(t,e);if(this.el=Bt.createElement(r,i),lt.currentNode=this.el.content,e===2){let d=this.el.content,u=d.firstChild;u.remove(),d.append(...u.childNodes)}for(;(o=lt.nextNode())!==null&&h.length<a;){if(o.nodeType===1){if(o.hasAttributes()){let d=[];for(let u of o.getAttributeNames())if(u.endsWith(ye)||u.startsWith(nt)){let c=l[s++];if(d.push(u),c!==void 0){let f=o.getAttribute(c.toLowerCase()+ye).split(nt),_=/([.?@])?(.*)/.exec(c);h.push({type:1,index:n,name:_[2],strings:f,ctor:_[1]==="."?ho:_[1]==="?"?co:_[1]==="@"?po:Jt})}else h.push({type:6,index:n})}for(let u of d)o.removeAttribute(u)}if(fi.test(o.tagName)){let d=o.textContent.split(nt),u=d.length-1;if(u>0){o.textContent=gt?gt.emptyScript:"";for(let c=0;c<u;c++)o.append(d[c],xt()),lt.nextNode(),h.push({type:2,index:++n});o.append(d[u],xt())}}}else if(o.nodeType===8)if(o.data===ci)h.push({type:2,index:n});else{let d=-1;for(;(d=o.data.indexOf(nt,d+1))!==-1;)h.push({type:7,index:n}),d+=nt.length-1}n++}}static createElement(t,e){let i=dt.createElement("template");return i.innerHTML=t,i}};function wt(t,e,i=t,o){var n,s,a,h;if(e===bt)return e;let r=o!==void 0?(n=i._$Co)===null||n===void 0?void 0:n[o]:i._$Cl,l=Et(e)?void 0:e._$litDirective$;return(r==null?void 0:r.constructor)!==l&&((s=r==null?void 0:r._$AO)===null||s===void 0||s.call(r,!1),l===void 0?r=void 0:(r=new l(t),r._$AT(t,i,o)),o!==void 0?((a=(h=i)._$Co)!==null&&a!==void 0?a:h._$Co=[])[o]=r:i._$Cl=r),r!==void 0&&(e=wt(t,r._$AS(t,e.values),r,o)),e}var lo=class{constructor(e,i){this._$AV=[],this._$AN=void 0,this._$AD=e,this._$AM=i}get parentNode(){return this._$AM.parentNode}get _$AU(){return this._$AM._$AU}u(e){var i;let{el:{content:o},parts:n}=this._$AD,s=((i=e==null?void 0:e.creationScope)!==null&&i!==void 0?i:dt).importNode(o,!0);lt.currentNode=s;let a=lt.nextNode(),h=0,r=0,l=n[0];for(;l!==void 0;){if(h===l.index){let d;l.type===2?d=new Zt(a,a.nextSibling,this,e):l.type===1?d=new l.ctor(a,l.name,l.strings,this,e):l.type===6&&(d=new fo(a,this,e)),this._$AV.push(d),l=n[++r]}h!==(l==null?void 0:l.index)&&(a=lt.nextNode(),h++)}return lt.currentNode=dt,s}v(e){let i=0;for(let o of this._$AV)o!==void 0&&(o.strings!==void 0?(o._$AI(e,o,i),i+=o.strings.length-2):o._$AI(e[i])),i++}},Zt=class{constructor(t,e,i,o){var n;this.type=2,this._$AH=U,this._$AN=void 0,this._$AA=t,this._$AB=e,this._$AM=i,this.options=o,this._$Cp=(n=o==null?void 0:o.isConnected)===null||n===void 0||n}get _$AU(){var t,e;return(e=(t=this._$AM)===null||t===void 0?void 0:t._$AU)!==null&&e!==void 0?e:this._$Cp}get parentNode(){let t=this._$AA.parentNode,e=this._$AM;return e!==void 0&&(t==null?void 0:t.nodeType)===11&&(t=e.parentNode),t}get startNode(){return this._$AA}get endNode(){return this._$AB}_$AI(t,e=this){t=wt(this,t,e),Et(t)?t===U||t==null||t===""?(this._$AH!==U&&this._$AR(),this._$AH=U):t!==this._$AH&&t!==bt&&this._(t):t._$litType$!==void 0?this.g(t):t.nodeType!==void 0?this.$(t):ro(t)?this.T(t):this._(t)}k(t){return this._$AA.parentNode.insertBefore(t,this._$AB)}$(t){this._$AH!==t&&(this._$AR(),this._$AH=this.k(t))}_(t){this._$AH!==U&&Et(this._$AH)?this._$AA.nextSibling.data=t:this.$(dt.createTextNode(t)),this._$AH=t}g(t){var e;let{values:i,_$litType$:o}=t,n=typeof o=="number"?this._$AC(t):(o.el===void 0&&(o.el=Bt.createElement(mi(o.h,o.h[0]),this.options)),o);if(((e=this._$AH)===null||e===void 0?void 0:e._$AD)===n)this._$AH.v(i);else{let s=new lo(n,this),a=s.u(this.options);s.v(i),this.$(a),this._$AH=s}}_$AC(t){let e=Xe.get(t.strings);return e===void 0&&Xe.set(t.strings,e=new Bt(t)),e}T(t){pi(this._$AH)||(this._$AH=[],this._$AR());let e=this._$AH,i,o=0;for(let n of t)o===e.length?e.push(i=new Zt(this.k(xt()),this.k(xt()),this,this.options)):i=e[o],i._$AI(n),o++;o<e.length&&(this._$AR(i&&i._$AB.nextSibling,o),e.length=o)}_$AR(t=this._$AA.nextSibling,e){var i;for((i=this._$AP)===null||i===void 0||i.call(this,!1,!0,e);t&&t!==this._$AB;){let o=t.nextSibling;t.remove(),t=o}}setConnected(t){var e;this._$AM===void 0&&(this._$Cp=t,(e=this._$AP)===null||e===void 0||e.call(this,t))}},Jt=class{constructor(t,e,i,o,n){this.type=1,this._$AH=U,this._$AN=void 0,this.element=t,this.name=e,this._$AM=o,this.options=n,i.length>2||i[0]!==""||i[1]!==""?(this._$AH=Array(i.length-1).fill(new String),this.strings=i):this._$AH=U}get tagName(){return this.element.tagName}get _$AU(){return this._$AM._$AU}_$AI(t,e=this,i,o){let n=this.strings,s=!1;if(n===void 0)t=wt(this,t,e,0),s=!Et(t)||t!==this._$AH&&t!==bt,s&&(this._$AH=t);else{let a=t,h,r;for(t=n[0],h=0;h<n.length-1;h++)r=wt(this,a[i+h],e,h),r===bt&&(r=this._$AH[h]),s||(s=!Et(r)||r!==this._$AH[h]),r===U?t=U:t!==U&&(t+=(r??"")+n[h+1]),this._$AH[h]=r}s&&!o&&this.j(t)}j(t){t===U?this.element.removeAttribute(this.name):this.element.setAttribute(this.name,t??"")}},ho=class extends Jt{constructor(){super(...arguments),this.type=3}j(e){this.element[this.name]=e===U?void 0:e}},uo=gt?gt.emptyScript:"",co=class extends Jt{constructor(){super(...arguments),this.type=4}j(e){e&&e!==U?this.element.setAttribute(this.name,uo):this.element.removeAttribute(this.name)}},po=class extends Jt{constructor(t,e,i,o,n){super(t,e,i,o,n),this.type=5}_$AI(t,e=this){var i;if((t=(i=wt(this,t,e,0))!==null&&i!==void 0?i:U)===bt)return;let o=this._$AH,n=t===U&&o!==U||t.capture!==o.capture||t.once!==o.once||t.passive!==o.passive,s=t!==U&&(o===U||n);n&&this.element.removeEventListener(this.name,this,o),s&&this.element.addEventListener(this.name,this,t),this._$AH=t}handleEvent(t){var e,i;typeof this._$AH=="function"?this._$AH.call((i=(e=this.options)===null||e===void 0?void 0:e.host)!==null&&i!==void 0?i:this.element,t):this._$AH.handleEvent(t)}},fo=class{constructor(e,i,o){this.element=e,this.type=6,this._$AN=void 0,this._$AM=i,this.options=o}get _$AU(){return this._$AM._$AU}_$AI(e){wt(this,e)}},ae=Ft.litHtmlPolyfillSupport;ae==null||ae(Bt,Zt),((re=Ft.litHtmlVersions)!==null&&re!==void 0?re:Ft.litHtmlVersions=[]).push("2.8.0");var mo=(t,e,i)=>{var o,n;let s=(o=i==null?void 0:i.renderBefore)!==null&&o!==void 0?o:e,a=s._$litPart$;if(a===void 0){let h=(n=i==null?void 0:i.renderBefore)!==null&&n!==void 0?n:null;s._$litPart$=a=new Zt(e.insertBefore(xt(),h),h,void 0,i??{})}return a._$AI(t),a},le,he,At=class extends mt{constructor(){super(...arguments),this.renderOptions={host:this},this._$Do=void 0}createRenderRoot(){var e,i;let o=super.createRenderRoot();return(e=(i=this.renderOptions).renderBefore)!==null&&e!==void 0||(i.renderBefore=o.firstChild),o}update(e){let i=this.render();this.hasUpdated||(this.renderOptions.isConnected=this.isConnected),super.update(e),this._$Do=mo(i,this.renderRoot,this.renderOptions)}connectedCallback(){var e;super.connectedCallback(),(e=this._$Do)===null||e===void 0||e.setConnected(!0)}disconnectedCallback(){var e;super.disconnectedCallback(),(e=this._$Do)===null||e===void 0||e.setConnected(!1)}render(){return bt}};At.finalized=!0,At._$litElement$=!0,(le=globalThis.litElementHydrateSupport)===null||le===void 0||le.call(globalThis,{LitElement:At});var de=globalThis.litElementPolyfillSupport;de==null||de({LitElement:At});((he=globalThis.litElementVersions)!==null&&he!==void 0?he:globalThis.litElementVersions=[]).push("3.3.3");var vo=eo`
  @font-face {
    font-family: 'Karla';
    font-weight: regular;
    src: url('./fonts/Karla-regular.woff') format('woff');
  }

  * {
    box-sizing: border-box;
  }

  :host {
    --lottie-player-toolbar-height: 35px;
    --lottie-player-toolbar-background-color: transparent;
    --lottie-player-toolbar-hover-background-color: #f3f6f8;
    --lottie-player-toolbar-icon-color: #20272c;
    --lottie-player-toolbar-icon-hover-color: #f3f6f8;
    --lottie-player-toolbar-icon-active-color: #00ddb3;
    --lottie-player-seeker-track-color: #00ddb3;
    --lottie-player-seeker-accent-color: #00c1a2;
    --lottie-player-seeker-thumb-color: #00c1a2;
    --lottie-player-options-separator: #d9e0e6;

    display: block;
    width: 100%;
    height: 100%;

    font-family: 'Karla', sans-serif;
    font-style: normal;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  .active {
    color: var(--lottie-player-toolbar-icon-active-color) !important;
  }

  .main {
    position: relative;
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
  }

  .animation {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
  }
  .animation.controls {
    height: calc(100% - var(--lottie-player-toolbar-height));
  }

  .toolbar {
    display: flex;
    align-items: center;
    justify-items: center;
    background-color: var(--lottie-player-toolbar-background-color);
    margin: 0 8px;
    height: var(--lottie-player-toolbar-height);
  }

  .btn-spacing-left {
    margin-right: 4px;
    margin-left: 8px;
  }

  .btn-spacing-center {
    margin-right: 4px;
    margin-left: 4px;
  }

  .btn-spacing-right {
    margin-right: 8px;
    margin-left: 4px;
  }

  .toolbar button {
    color: #20272c;
    cursor: pointer;
    fill: var(--lottie-player-toolbar-icon-color);
    display: flex;
    background: none;
    border: 0px;
    border-radius: 4px;
    padding: 4px;
    outline: none;
    width: 24px;
    height: 24px;
    align-items: center;
  }

  .toolbar button:hover {
    background-color: var(--lottie-player-toolbar-icon-hover-color);
    border-style: solid;
    border-radius: 2px;
  }

  .toolbar button.active {
    fill: var(--lottie-player-toolbar-icon-active-color);
  }

  .toolbar button.active:hover {
    fill: var(--lottie-player-toolbar-icon-hover-color);
    border-radius: 4px;
  }

  .toolbar button:focus-visible {
    outline: 2px solid var(--lottie-player-toolbar-icon-active-color);
    border-radius: 4px;
    box-sizing: border-box;
  }

  .toolbar button svg {
    width: 16px;
    height: 16px;
  }

  .toolbar button.disabled svg {
    display: none;
  }

  .popover {
    position: absolute;
    bottom: 40px;
    left: calc(100% - 239px);
    width: 224px;
    min-height: 84px;
    max-height: 300px;
    background-color: #ffffff;
    box-shadow: 0px 8px 48px 0px rgba(243, 246, 248, 0.15), 0px 8px 16px 0px rgba(61, 72, 83, 0.16),
      0px 0px 1px 0px rgba(61, 72, 83, 0.36);
    border-radius: 8px;
    padding: 8px;
    z-index: 100;
    overflow-y: scroll;
    scrollbar-width: none;
  }
  .popover:focus-visible {
    outline: 2px solid var(--lottie-player-toolbar-icon-active-color);
    border-radius: 4px;
    box-sizing: border-box;
  }

  .popover::-webkit-scrollbar {
    width: 0px;
  }

  .popover-button {
    background: none;
    border: none;
    font-family: inherit;
    width: 100%;
    flex-direction: row;
    cursor: pointer;
    height: 32px;
    color: #20272c;
    justify-content: space-between;
    display: flex;
    padding: 4px 8px;
    align-items: flex-start;
    gap: 8px;
    align-self: stretch;
    border-radius: 4px;
  }

  .popover-button:focus-visible {
    outline: 2px solid var(--lottie-player-toolbar-icon-active-color);
    border-radius: 4px;
    box-sizing: border-box;
  }

  .popover-button:hover {
    background-color: var(--lottie-player-toolbar-hover-background-color);
  }

  .popover-button-text {
    display: flex;
    color: #20272c;
    flex-direction: column;
    align-self: stretch;
    justify-content: center;
    font-family: inherit;
    font-size: 14px;
    font-style: normal;
    font-weight: 400;
    line-height: 150%;
    letter-spacing: -0.28px;
  }

  .reset-btn {
    font-size: 12px;
    cursor: pointer;
    font-family: inherit;
    background: none;
    border: none;
    font-weight: 400;
    line-height: 18px;
    letter-spacing: 0em;
    text-align: left;
    color: #63727e;
    padding: 0;
    width: 31px;
    height: 18px;
  }
  .reset-btn:focus-visible {
    outline: 2px solid var(--lottie-player-toolbar-icon-active-color);
    border-radius: 4px;
    box-sizing: border-box;
  }
  .reset-btn:hover {
    color: #20272c;
  }

  .option-title-button {
    display: flex;
    flex-direction: row;
    width: 100%;
    height: 32px;
    align-items: center;
    gap: 4px;
    align-self: stretch;
    cursor: pointer;
    color: var(--lottie-player-toolbar-icon-color);
    border: none;
    background: none;
    padding: 4px;
    font-family: inherit;
    font-size: 16px;
    font-weight: 700;
    line-height: 150%;
    letter-spacing: -0.32px;
  }
  .option-title-button.themes {
    width: auto;
    padding: 0;
  }
  .option-title-button:hover {
    background-color: var(--lottie-player-toolbar-icon-hover-color);
  }

  .option-title-themes-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1 0 0;
  }
  .option-title-themes-row:hover {
    background-color: var(--lottie-player-toolbar-icon-hover-color);
  }

  .option-title-button:focus-visible {
    outline: 2px solid var(--lottie-player-toolbar-icon-active-color);
    border-radius: 4px;
    box-sizing: border-box;
  }

  .option-title-text {
    font-size: 16px;
    font-style: normal;
    font-weight: 700;
    line-height: 150%;
    letter-spacing: -0.32px;
  }

  .option-title-separator {
    margin: 8px -8px;
    border-bottom: 1px solid var(--lottie-player-options-separator);
  }

  .option-title-chevron {
    display: flex;
    padding: 4px;
    border-radius: 8px;
    justify-content: center;
    align-items: center;
    gap: 8px;
  }

  .option-row {
    display: flex;
    flex-direction: column;
  }
  .option-row > ul {
    padding: 0;
    margin: 0;
  }

  .option-button {
    width: 100%;
    background: none;
    border: none;
    font-family: inherit;
    display: flex;
    padding: 4px 8px;
    color: #20272c;
    overflow: hidden;
    align-items: center;
    gap: 8px;
    align-self: stretch;
    cursor: pointer;
    height: 32px;
    font-family: inherit;
    font-size: 14px;
    border-radius: 4px;
  }
  .option-button:hover {
    background-color: var(--lottie-player-toolbar-hover-background-color);
  }
  .option-button:focus-visible {
    outline: 2px solid var(--lottie-player-toolbar-icon-active-color);
    border-radius: 4px;
    box-sizing: border-box;
  }

  .option-tick {
    display: flex;
    width: 24px;
    height: 24px;
    align-items: flex-start;
    gap: 8px;
  }

  .seeker {
    height: 4px;
    width: 95%;
    outline: none;
    -webkit-appearance: none;
    -moz-apperance: none;
    border-radius: 9999px;
    cursor: pointer;
    background-image: linear-gradient(
      to right,
      rgb(0, 221, 179) calc(var(--seeker) * 1%),
      rgb(217, 224, 230) calc(var(--seeker) * 1%)
    );
  }
  .seeker.to-left {
    background-image: linear-gradient(
      to right,
      rgb(217, 224, 230) calc(var(--seeker) * 1%),
      rgb(0, 221, 179) calc(var(--seeker) * 1%)
    );
  }
  .seeker::-webkit-slider-runnable-track:focus-visible {
    color: #f07167;
    accent-color: #00ddb3;
  }

  .seeker::-webkit-slider-runnable-track {
    width: 100%;
    height: 5px;
    cursor: pointer;
  }
  .seeker::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 16px;
    width: 16px;
    border-radius: 50%;
    background: var(--lottie-player-seeker-thumb-color);
    cursor: pointer;
    margin-top: -5px;
  }
  .seeker:focus-visible::-webkit-slider-thumb {
    background: var(--lottie-player-seeker-thumb-color);
    outline: 2px solid var(--lottie-player-seeker-track-color);
    border: 1.5px solid #ffffff;
  }
  .seeker::-webkit-slider-thumb:hover {
    background: #019d91;
  }
  .seeker::-moz-range-thumb {
    appearance: none;
    height: 16px;
    width: 16px;
    border-radius: 50%;
    background: var(--lottie-player-seeker-thumb-color);
    cursor: pointer;
    margin-top: -5px;
    border-color: transparent;
  }
  .seeker:focus-visible::-moz-range-thumb {
    background: var(--lottie-player-seeker-thumb-color);
    outline: 2px solid var(--lottie-player-seeker-track-color);
    border: 1.5px solid #ffffff;
  }

  .error {
    display: flex;
    justify-content: center;
    margin: auto;
    height: 100%;
    align-items: center;
  }
`;/*! Bundled license information:

@lit/reactive-element/css-tag.js:
  (**
   * @license
   * Copyright 2019 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/reactive-element.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-html/lit-html.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-element/lit-element.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-html/is-server.js:
  (**
   * @license
   * Copyright 2022 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)
*/var Ge={},_o=function(t,e,i,o,n){var s=new Worker(Ge[e]||(Ge[e]=URL.createObjectURL(new Blob([t+';addEventListener("error",function(e){e=e.error;postMessage({$e$:[e.message,e.code,e.stack]})})'],{type:"text/javascript"}))));return s.onmessage=function(a){var h=a.data,r=h.$e$;if(r){var l=new Error(r[0]);l.code=r[1],l.stack=r[2],n(l,null)}else n(null,h)},s.postMessage(i,o),s},V=Uint8Array,ht=Uint16Array,vi=Int32Array,Ee=new V([0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0,0]),Ie=new V([0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,0,0]),_i=new V([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]),yi=function(t,e){for(var i=new ht(31),o=0;o<31;++o)i[o]=e+=1<<t[o-1];for(var n=new vi(i[30]),o=1;o<30;++o)for(var s=i[o];s<i[o+1];++s)n[s]=s-i[o]<<5|o;return{b:i,r:n}},gi=yi(Ee,2),Oe=gi.b,yo=gi.r;Oe[28]=258,yo[258]=28;var go=yi(Ie,0),bi=go.b,Ht=new ht(32768);for(L=0;L<32768;++L)et=(L&43690)>>1|(L&21845)<<1,et=(et&52428)>>2|(et&13107)<<2,et=(et&61680)>>4|(et&3855)<<4,Ht[L]=((et&65280)>>8|(et&255)<<8)>>1;var et,L,_t=function(t,e,i){for(var o=t.length,n=0,s=new ht(e);n<o;++n)t[n]&&++s[t[n]-1];var a=new ht(e);for(n=1;n<e;++n)a[n]=a[n-1]+s[n-1]<<1;var h;if(i){h=new ht(1<<e);var r=15-e;for(n=0;n<o;++n)if(t[n])for(var l=n<<4|t[n],d=e-t[n],u=a[t[n]-1]++<<d,c=u|(1<<d)-1;u<=c;++u)h[Ht[u]>>r]=l}else for(h=new ht(o),n=0;n<o;++n)t[n]&&(h[n]=Ht[a[t[n]-1]++]>>15-t[n]);return h},It=new V(288);for(L=0;L<144;++L)It[L]=8;var L;for(L=144;L<256;++L)It[L]=9;var L;for(L=256;L<280;++L)It[L]=7;var L;for(L=280;L<288;++L)It[L]=8;var L,wi=new V(32);for(L=0;L<32;++L)wi[L]=5;var L,Ci=_t(It,9,1),Li=_t(wi,5,1),zt=function(t){for(var e=t[0],i=1;i<t.length;++i)t[i]>e&&(e=t[i]);return e},R=function(t,e,i){var o=e/8|0;return(t[o]|t[o+1]<<8)>>(e&7)&i},Dt=function(t,e){var i=e/8|0;return(t[i]|t[i+1]<<8|t[i+2]<<16)>>(e&7)},Si=function(t){return(t+7)/8|0},Wt=function(t,e,i){(e==null||e<0)&&(e=0),(i==null||i>t.length)&&(i=t.length);var o=new V(i-e);return o.set(t.subarray(e,i)),o},Ai=["unexpected EOF","invalid block type","invalid length/literal","invalid distance","stream finished","no stream handler",,"no callback","invalid UTF-8 data","extra field too long","date not in range 1980-2099","filename too long","stream finishing","invalid zip data"],D=function(t,e,i){var o=new Error(e||Ai[t]);if(o.code=t,Error.captureStackTrace&&Error.captureStackTrace(o,D),!i)throw o;return o},$i=function(t,e,i,o){var n=t.length,s=o?o.length:0;if(!n||e.f&&!e.l)return i||new V(0);var a=!i||e.i!=2,h=e.i;i||(i=new V(n*3));var r=function(Ve){var Fe=i.length;if(Ve>Fe){var Be=new V(Math.max(Fe*2,Ve));Be.set(i),i=Be}},l=e.f||0,d=e.p||0,u=e.b||0,c=e.l,f=e.d,_=e.m,m=e.n,v=n*8;do{if(!c){l=R(t,d,1);var g=R(t,d+1,3);if(d+=3,g)if(g==1)c=Ci,f=Li,_=9,m=5;else if(g==2){var y=R(t,d,31)+257,b=R(t,d+10,15)+4,S=y+R(t,d+5,31)+1;d+=14;for(var $=new V(S),I=new V(19),k=0;k<b;++k)I[_i[k]]=R(t,d+k*3,7);d+=b*3;for(var F=zt(I),O=(1<<F)-1,tt=_t(I,F,1),k=0;k<S;){var B=tt[R(t,d,O)];d+=B&15;var M=B>>4;if(M<16)$[k++]=M;else{var it=0,st=0;for(M==16?(st=3+R(t,d,3),d+=2,it=$[k-1]):M==17?(st=3+R(t,d,7),d+=3):M==18&&(st=11+R(t,d,127),d+=7);st--;)$[k++]=it}}var J=$.subarray(0,y),j=$.subarray(y);_=zt(J),m=zt(j),c=_t(J,_,1),f=_t(j,m,1)}else D(1);else{var M=Si(d)+4,Yt=t[M-4]|t[M-3]<<8,Xt=M+Yt;if(Xt>n){h&&D(0);break}a&&r(u+Yt),i.set(t.subarray(M,Xt),u),e.b=u+=Yt,e.p=d=Xt*8,e.f=l;continue}if(d>v){h&&D(0);break}}a&&r(u+131072);for(var Bi=(1<<_)-1,Hi=(1<<m)-1,Gt=d;;Gt=d){var it=c[Dt(t,d)&Bi],pt=it>>4;if(d+=it&15,d>v){h&&D(0);break}if(it||D(2),pt<256)i[u++]=pt;else if(pt==256){Gt=d,c=null;break}else{var ze=pt-254;if(pt>264){var k=pt-257,Lt=Ee[k];ze=R(t,d,(1<<Lt)-1)+Oe[k],d+=Lt}var Qt=f[Dt(t,d)&Hi],te=Qt>>4;Qt||D(3),d+=Qt&15;var j=bi[te];if(te>3){var Lt=Ie[te];j+=Dt(t,d)&(1<<Lt)-1,d+=Lt}if(d>v){h&&D(0);break}a&&r(u+131072);var ee=u+ze;if(u<j){var De=s-j,qi=Math.min(j,ee);for(De+u<0&&D(3);u<qi;++u)i[u]=o[De+u]}for(;u<ee;u+=4)i[u]=i[u-j],i[u+1]=i[u+1-j],i[u+2]=i[u+2-j],i[u+3]=i[u+3-j];u=ee}}e.l=c,e.p=Gt,e.b=u,e.f=l,c&&(l=1,e.m=_,e.d=f,e.n=m)}while(!l);return u==i.length?i:Wt(i,0,u)},bo=new V(0),wo=function(t,e){var i={};for(var o in t)i[o]=t[o];for(var o in e)i[o]=e[o];return i},Qe=function(t,e,i){for(var o=t(),n=t.toString(),s=n.slice(n.indexOf("[")+1,n.lastIndexOf("]")).replace(/\s+/g,"").split(","),a=0;a<o.length;++a){var h=o[a],r=s[a];if(typeof h=="function"){e+=";"+r+"=";var l=h.toString();if(h.prototype)if(l.indexOf("[native code]")!=-1){var d=l.indexOf(" ",8)+1;e+=l.slice(d,l.indexOf("(",d))}else{e+=l;for(var u in h.prototype)e+=";"+r+".prototype."+u+"="+h.prototype[u].toString()}else e+=l}else i[r]=h}return e},jt=[],Co=function(t){var e=[];for(var i in t)t[i].buffer&&e.push((t[i]=new t[i].constructor(t[i])).buffer);return e},Lo=function(t,e,i,o){if(!jt[i]){for(var n="",s={},a=t.length-1,h=0;h<a;++h)n=Qe(t[h],n,s);jt[i]={c:Qe(t[a],n,s),e:s}}var r=wo({},jt[i].e);return _o(jt[i].c+";onmessage=function(e){for(var k in e.data)self[k]=e.data[k];onmessage="+e.toString()+"}",i,r,Co(r),o)},So=function(){return[V,ht,vi,Ee,Ie,_i,Oe,bi,Ci,Li,Ht,Ai,_t,zt,R,Dt,Si,Wt,D,$i,Me,ki,xi]},ki=function(t){return postMessage(t,[t.buffer])},xi=function(t){return t&&{out:t.size&&new V(t.size),dictionary:t.dictionary}},Ao=function(t,e,i,o,n,s){var a=Lo(i,o,n,function(h,r){a.terminate(),s(h,r)});return a.postMessage([t,e],e.consume?[t.buffer]:[]),function(){a.terminate()}},G=function(t,e){return t[e]|t[e+1]<<8},K=function(t,e){return(t[e]|t[e+1]<<8|t[e+2]<<16|t[e+3]<<24)>>>0},ue=function(t,e){return K(t,e)+K(t,e+4)*4294967296};function $o(t,e,i){return i||(i=e,e={}),typeof i!="function"&&D(7),Ao(t,e,[So],function(o){return ki(Me(o.data[0],xi(o.data[1])))},1,i)}function Me(t,e){return $i(t,{i:2},e&&e.out,e&&e.dictionary)}var ge=typeof TextDecoder<"u"&&new TextDecoder,ko=0;try{ge.decode(bo,{stream:!0}),ko=1}catch{}var xo=function(t){for(var e="",i=0;;){var o=t[i++],n=(o>127)+(o>223)+(o>239);if(i+n>t.length)return{s:e,r:Wt(t,i-1)};n?n==3?(o=((o&15)<<18|(t[i++]&63)<<12|(t[i++]&63)<<6|t[i++]&63)-65536,e+=String.fromCharCode(55296|o>>10,56320|o&1023)):n&1?e+=String.fromCharCode((o&31)<<6|t[i++]&63):e+=String.fromCharCode((o&15)<<12|(t[i++]&63)<<6|t[i++]&63):e+=String.fromCharCode(o)}};function Ct(t,e){if(e){for(var i="",o=0;o<t.length;o+=16384)i+=String.fromCharCode.apply(null,t.subarray(o,o+16384));return i}else{if(ge)return ge.decode(t);var n=xo(t),s=n.s,i=n.r;return i.length&&D(8),s}}var Eo=function(t,e){return e+30+G(t,e+26)+G(t,e+28)},Io=function(t,e,i){var o=G(t,e+28),n=Ct(t.subarray(e+46,e+46+o),!(G(t,e+8)&2048)),s=e+46+o,a=K(t,e+20),h=i&&a==4294967295?Oo(t,s):[a,K(t,e+24),K(t,e+42)],r=h[0],l=h[1],d=h[2];return[G(t,e+10),r,l,n,s+G(t,e+30)+G(t,e+32),d]},Oo=function(t,e){for(;G(t,e)!=1;e+=4+G(t,e+2));return[ue(t,e+12),ue(t,e+4),ue(t,e+20)]},ti=typeof queueMicrotask=="function"?queueMicrotask:typeof setTimeout=="function"?setTimeout:function(t){t()};function Mo(t,e,i){i||(i=e,e={}),typeof i!="function"&&D(7);var o=[],n=function(){for(var v=0;v<o.length;++v)o[v]()},s={},a=function(v,g){ti(function(){i(v,g)})};ti(function(){a=i});for(var h=t.length-22;K(t,h)!=101010256;--h)if(!h||t.length-h>65558)return a(D(13,0,1),null),n;var r=G(t,h+8);if(r){var l=r,d=K(t,h+16),u=d==4294967295||l==65535;if(u){var c=K(t,h-12);u=K(t,c)==101075792,u&&(l=r=K(t,c+32),d=K(t,c+48))}for(var f=e&&e.filter,_=function(v){var g=Io(t,d,u),y=g[0],b=g[1],S=g[2],$=g[3],I=g[4],k=g[5],F=Eo(t,k);d=I;var O=function(B,M){B?(n(),a(B,null)):(M&&(s[$]=M),--r||a(null,s))};if(!f||f({name:$,size:b,originalSize:S,compression:y}))if(!y)O(null,Wt(t,F,F+b));else if(y==8){var tt=t.subarray(F,F+b);if(b<32e4)try{O(null,Me(tt,{out:new V(S)}))}catch(B){O(B,null)}else o.push($o(tt,{size:S},O))}else O(D(14,"unknown compression type "+y,1),null);else O(null,null)},m=0;m<l;++m)_(m)}else a(null,{});return n}function Po(t){return(Array.isArray(t)?t:t.issues).reduce((e,i)=>{if(i.path){let o=i.path.map(({key:n})=>n).join(".");e.nested[o]=[...e.nested[o]||[],i.message]}else e.root=[...e.root||[],i.message];return e},{nested:{}})}var To=class extends Error{constructor(e){super(e[0].message);p(this,"issues");this.name="ValiError",this.issues=e}};function Uo(t,e){return{reason:t==null?void 0:t.reason,validation:e.validation,origin:(t==null?void 0:t.origin)||"value",message:e.message,input:e.input,abortEarly:t==null?void 0:t.abortEarly,abortPipeEarly:t==null?void 0:t.abortPipeEarly}}function jo(t,e){return{reason:e,origin:t==null?void 0:t.origin,abortEarly:t==null?void 0:t.abortEarly,abortPipeEarly:t==null?void 0:t.abortPipeEarly}}function rt(t,e,i,o){if(!e||!e.length)return{output:t};let n,s,a=t;for(let h of e){let r=h(a);if(r.issue){n=n||jo(i,o);let l=Uo(n,r.issue);if(s?s.push(l):s=[l],n.abortEarly||n.abortPipeEarly)break}else a=r.output}return s?{issues:s}:{output:a}}function X(t,e){return!t||typeof t=="string"?[t,e]:[void 0,t]}function Q(t,e,i,o,n,s){return{issues:[{reason:e,validation:i,origin:(t==null?void 0:t.origin)||"value",message:o,input:n,issues:s,abortEarly:t==null?void 0:t.abortEarly,abortPipeEarly:t==null?void 0:t.abortPipeEarly}]}}function No(t=[]){return{schema:"any",async:!1,_parse(e,i){return rt(e,t,i,"any")}}}function $t(t,e,i){let[o,n]=X(e,i);return{schema:"array",array:{item:t},async:!1,_parse(s,a){if(!Array.isArray(s))return Q(a,"type","array",o||"Invalid type",s);let h,r=[];for(let l=0;l<s.length;l++){let d=s[l],u=t._parse(d,a);if(u.issues){let c={schema:"array",input:s,key:l,value:d};for(let f of u.issues)f.path?f.path.unshift(c):f.path=[c],h==null||h.push(f);if(h||(h=u.issues),a==null?void 0:a.abortEarly)break}else r.push(u.output)}return h?{issues:h}:rt(r,n,a,"array")}}}function ce(t,e){let[i,o]=X(t,e);return{schema:"boolean",async:!1,_parse(n,s){return typeof n!="boolean"?Q(s,"type","boolean",i||"Invalid type",n):rt(n,o,s,"boolean")}}}function ei(t,e){return{schema:"literal",literal:t,async:!1,_parse(i,o){return i!==t?Q(o,"type","literal",e||"Invalid type",i):{output:i}}}}function zo(t,e){return{schema:"native_enum",nativeEnum:t,async:!1,_parse(i,o){return Object.values(t).includes(i)?{output:i}:Q(o,"type","native_enum",e||"Invalid type",i)}}}function Y(t,e){let[i,o]=X(t,e);return{schema:"number",async:!1,_parse(n,s){return typeof n!="number"?Q(s,"type","number",i||"Invalid type",n):rt(n,o,s,"number")}}}function Z(t,e,i){let[o,n]=X(e,i),s;return{schema:"object",object:t,async:!1,_parse(a,h){if(!a||typeof a!="object")return Q(h,"type","object",o||"Invalid type",a);s=s||Object.entries(t);let r,l={};for(let[d,u]of s){let c=a[d],f=u._parse(c,h);if(f.issues){let _={schema:"object",input:a,key:d,value:c};for(let m of f.issues)m.path?m.path.unshift(_):m.path=[_],r==null||r.push(m);if(r||(r=f.issues),h==null?void 0:h.abortEarly)break}else l[d]=f.output}return r?{issues:r}:rt(l,n,h,"object")}}}function A(t){return{schema:"optional",wrapped:t,async:!1,_parse(e,i){return e===void 0?{output:e}:t._parse(e,i)}}}function z(t,e){let[i,o]=X(t,e);return{schema:"string",async:!1,_parse(n,s){return typeof n!="string"?Q(s,"type","string",i||"Invalid type",n):rt(n,o,s,"string")}}}function Do(t,e,i,o){if(typeof e=="object"&&!Array.isArray(e)){let[a,h]=X(i,o);return[t,e,a,h]}let[n,s]=X(e,i);return[z(),t,n,s]}var Vo=["__proto__","prototype","constructor"];function Fo(t,e,i,o){let[n,s,a,h]=Do(t,e,i,o);return{schema:"record",record:{key:n,value:s},async:!1,_parse(r,l){if(!r||typeof r!="object")return Q(l,"type","record",a||"Invalid type",r);let d,u={};for(let[c,f]of Object.entries(r))if(!Vo.includes(c)){let _,m=n._parse(c,{origin:"key",abortEarly:l==null?void 0:l.abortEarly,abortPipeEarly:l==null?void 0:l.abortPipeEarly});if(m.issues){_={schema:"record",input:r,key:c,value:f};for(let g of m.issues)g.path=[_],d==null||d.push(g);if(d||(d=m.issues),l==null?void 0:l.abortEarly)break}let v=s._parse(f,l);if(v.issues){_=_||{schema:"record",input:r,key:c,value:f};for(let g of v.issues)g.path?g.path.unshift(_):g.path=[_],d==null||d.push(g);if(d||(d=v.issues),l==null?void 0:l.abortEarly)break}!m.issues&&!v.issues&&(u[m.output]=v.output)}return d?{issues:d}:rt(u,h,l,"record")}}}function Bo(t,e,i){if(typeof t=="object"&&!Array.isArray(t)){let[s,a]=X(e,i);return[t,s,a]}let[o,n]=X(t,e);return[void 0,o,n]}function ii(t,e,i,o){let[n,s,a]=Bo(e,i,o);return{schema:"tuple",tuple:{items:t,rest:n},async:!1,_parse(h,r){if(!Array.isArray(h)||!n&&t.length!==h.length||n&&t.length>h.length)return Q(r,"type","tuple",s||"Invalid type",h);let l,d=[];for(let u=0;u<t.length;u++){let c=h[u],f=t[u]._parse(c,r);if(f.issues){let _={schema:"tuple",input:h,key:u,value:c};for(let m of f.issues)m.path?m.path.unshift(_):m.path=[_],l==null||l.push(m);if(l||(l=f.issues),r==null?void 0:r.abortEarly)break}else d[u]=f.output}if(n)for(let u=t.length;u<h.length;u++){let c=h[u],f=n._parse(c,r);if(f.issues){let _={schema:"tuple",input:h,key:u,value:c};for(let m of f.issues)m.path?m.path.unshift(_):m.path=[_],l==null||l.push(m);if(l||(l=f.issues),r==null?void 0:r.abortEarly)break}else d[u]=f.output}return l?{issues:l}:rt(d,a,r,"tuple")}}}function be(t,e){return{schema:"union",union:t,async:!1,_parse(i,o){let n,s;for(let a of t){let h=a._parse(i,o);if(h.issues)if(n)for(let r of h.issues)n.push(r);else n=h.issues;else{s=[h.output];break}}return s?{output:s[0]}:Q(o,"type","union",e||"Invalid type",i,n)}}}function Ot(t,e,i){let[o,n]=X(e,i);return Z(t.reduce((s,a)=>({...s,...a.object}),{}),o,n)}function Ho(t,e,i,o){let[n,s]=X(i,o);return Z(Object.entries(t.object).reduce((a,[h,r])=>e.includes(h)?a:{...a,[h]:r},{}),n,s)}function qo(t,e,i){let o=t._parse(e,i);return o.issues?{success:!1,error:new To(o.issues),issues:o.issues}:{success:!0,data:o.output,output:o.output}}function we(t,e){return i=>i>t?{issue:{validation:"max_value",message:e||"Invalid value",input:i}}:{output:i}}function Ce(t,e){return i=>i<t?{issue:{validation:"min_value",message:e||"Invalid value",input:i}}:{output:i}}var Ro=Object.create,Pe=Object.defineProperty,Zo=Object.getOwnPropertyDescriptor,Ei=Object.getOwnPropertyNames,Jo=Object.getPrototypeOf,Wo=Object.prototype.hasOwnProperty,Ko=(t,e,i)=>e in t?Pe(t,e,{enumerable:!0,configurable:!0,writable:!0,value:i}):t[e]=i,Mt=(t,e)=>function(){return e||(0,t[Ei(t)[0]])((e={exports:{}}).exports,e),e.exports},Yo=(t,e,i,o)=>{if(e&&typeof e=="object"||typeof e=="function")for(let n of Ei(e))!Wo.call(t,n)&&n!==i&&Pe(t,n,{get:()=>e[n],enumerable:!(o=Zo(e,n))||o.enumerable});return t},Xo=(t,e,i)=>(i=t!=null?Ro(Jo(t)):{},Yo(e||!t||!t.__esModule?Pe(i,"default",{value:t,enumerable:!0}):i,t)),Go=(t,e,i)=>(Ko(t,typeof e!="symbol"?e+"":e,i),i),Qo=Mt({"../../node_modules/.pnpm/@rgba-image+copy@0.1.3/node_modules/@rgba-image/copy/dist/index.js"(t){Object.defineProperty(t,"__esModule",{value:!0}),t.copy=void 0;var e=(i,o,n=0,s=0,a=i.width-n,h=i.height-s,r=0,l=0)=>{if(n=n|0,s=s|0,a=a|0,h=h|0,r=r|0,l=l|0,a<=0||h<=0)return;let d=new Uint32Array(i.data.buffer),u=new Uint32Array(o.data.buffer);for(let c=0;c<h;c++){let f=s+c;if(f<0||f>=i.height)continue;let _=l+c;if(!(_<0||_>=o.height))for(let m=0;m<a;m++){let v=n+m;if(v<0||v>=i.width)continue;let g=r+m;if(g<0||g>=o.width)continue;let y=f*i.width+v,b=_*o.width+g;u[b]=d[y]}}};t.copy=e}}),tn=Mt({"../../node_modules/.pnpm/@rgba-image+create-image@0.1.1/node_modules/@rgba-image/create-image/dist/index.js"(t){Object.defineProperty(t,"__esModule",{value:!0}),t.CreateImageFactory=(e=[0,0,0,0],i=4)=>{if(i=Math.floor(i),isNaN(i)||i<1)throw TypeError("channels should be a positive non-zero number");if(!("length"in e)||e.length<i)throw TypeError(`fill should be iterable with at least ${i} members`);e=new Uint8ClampedArray(e).slice(0,i);let o=e.every(n=>n===0);return(n,s,a)=>{if(n===void 0||s===void 0)throw TypeError("Not enough arguments");if(n=Math.floor(n),s=Math.floor(s),isNaN(n)||n<1||isNaN(s)||s<1)throw TypeError("Index or size is negative or greater than the allowed amount");let h=n*s*i;if(a===void 0&&(a=new Uint8ClampedArray(h)),a instanceof Uint8ClampedArray){if(a.length!==h)throw TypeError("Index or size is negative or greater than the allowed amount");if(!o)for(let r=0;r<s;r++)for(let l=0;l<n;l++){let d=(r*n+l)*i;for(let u=0;u<i;u++)a[d+u]=e[u]}return{get width(){return n},get height(){return s},get data(){return a}}}throw TypeError("Expected data to be Uint8ClampedArray or undefined")}},t.createImage=t.CreateImageFactory()}}),en=Mt({"../../node_modules/.pnpm/@rgba-image+lanczos@0.1.1/node_modules/@rgba-image/lanczos/dist/filters.js"(t){Object.defineProperty(t,"__esModule",{value:!0}),t.filters=void 0;var e=14,i=(s,a)=>{if(s<=-a||s>=a||s==0)return 0;let h=s*Math.PI;return Math.sin(h)/h*Math.sin(h/a)/(h/a)},o=s=>Math.round(s*((1<<e)-1)),n=(s,a,h,r,l)=>{let d=l?2:3,u=1/h,c=Math.min(1,h),f=d/c,_=Math.floor((f+1)*2),m=new Int16Array((_+2)*a),v=0;for(let g=0;g<a;g++){let y=(g+.5)*u+r,b=Math.max(0,Math.floor(y-f)),S=Math.min(s-1,Math.ceil(y+f)),$=S-b+1,I=new Float32Array($),k=new Int16Array($),F=0,O=0;for(let J=b;J<=S;J++){let j=i((J+.5-y)*c,d);F+=j,I[O]=j,O++}let tt=0;for(let J=0;J<I.length;J++){let j=I[J]/F;tt+=j,k[J]=o(j)}k[a>>1]+=o(1-tt);let B=0;for(;B<k.length&&k[B]===0;)B++;let M=k.length-1;for(;M>0&&k[M]===0;)M--;let it=b+B,st=M-B+1;m[v++]=it,m[v++]=st,m.set(k.subarray(B,M+1),v),v+=st}return m};t.filters=n}}),on=Mt({"../../node_modules/.pnpm/@rgba-image+lanczos@0.1.1/node_modules/@rgba-image/lanczos/dist/convolve.js"(t){Object.defineProperty(t,"__esModule",{value:!0}),t.convolve=void 0;var e=14,i=(o,n,s,a,h,r)=>{let l=0,d=0;for(let u=0;u<a;u++){let c=0;for(let f=0;f<h;f++){let _=r[c++],m=l+_*4|0,v=0,g=0,y=0,b=0;for(let S=r[c++];S>0;S--){let $=r[c++];v=v+$*o[m]|0,g=g+$*o[m+1]|0,y=y+$*o[m+2]|0,b=b+$*o[m+3]|0,m=m+4|0}n[d]=v+8192>>e,n[d+1]=g+8192>>e,n[d+2]=y+8192>>e,n[d+3]=b+8192>>e,d=d+a*4|0}d=(u+1)*4|0,l=(u+1)*s*4|0}};t.convolve=i}}),nn=Mt({"../../node_modules/.pnpm/@rgba-image+lanczos@0.1.1/node_modules/@rgba-image/lanczos/dist/index.js"(t){Object.defineProperty(t,"__esModule",{value:!0}),t.lanczos2=t.lanczos=void 0;var e=Qo(),i=tn(),o=en(),n=on(),s=(r,l,d=!1)=>{let u=l.width/r.width,c=l.height/r.height,f=o.filters(r.width,l.width,u,0,d),_=o.filters(r.height,l.height,c,0,d),m=new Uint8ClampedArray(l.width*r.height*4);n.convolve(r.data,m,r.width,r.height,l.width,f),n.convolve(m,l.data,r.height,l.width,l.height,_)},a=(r,l,d=0,u=0,c=r.width-d,f=r.height-u,_=0,m=0,v=l.width-_,g=l.height-m)=>{if(d=d|0,u=u|0,c=c|0,f=f|0,_=_|0,m=m|0,v=v|0,g=g|0,c<=0||f<=0||v<=0||g<=0)return;if(d===0&&u===0&&c===r.width&&f===r.height&&_===0&&m===0&&v===l.width&&g===l.height){s(r,l);return}let y=i.createImage(c,f),b=i.createImage(v,g);e.copy(r,y,d,u),s(y,b),e.copy(b,l,0,0,b.width,b.height,_,m)};t.lanczos=a;var h=(r,l,d=0,u=0,c=r.width-d,f=r.height-u,_=0,m=0,v=l.width-_,g=l.height-m)=>{if(d=d|0,u=u|0,c=c|0,f=f|0,_=_|0,m=m|0,v=v|0,g=g|0,c<=0||f<=0||v<=0||g<=0)return;if(d===0&&u===0&&c===r.width&&f===r.height&&_===0&&m===0&&v===l.width&&g===l.height){s(r,l,!0);return}let y=i.createImage(c,f),b=i.createImage(v,g);e.copy(r,y,d,u),s(y,b,!0),e.copy(b,l,0,0,b.width,b.height,_,m)};t.lanczos2=h}}),Ii=(t=>(t.Bounce="bounce",t.Normal="normal",t))(Ii||{}),rn=zo(Ii),Oi=Z({autoplay:A(ce()),defaultTheme:A(z()),direction:A(be([ei(1),ei(-1)])),hover:A(ce()),id:z(),intermission:A(Y()),loop:A(be([ce(),Y()])),playMode:A(rn),speed:A(Y()),themeColor:A(z())}),sn=Z({animations:$t(z()),id:z()}),an=Z({activeAnimationId:A(z()),animations:$t(Oi),author:A(z()),custom:A(Fo(z(),No())),description:A(z()),generator:A(z()),keywords:A(z()),revision:A(Y()),themes:A($t(sn)),states:A($t(z())),version:A(z())}),Mi=Ho(Oi,["id"]),ut=Z({state:z()}),ln=ut,hn=Ot([ut,Z({ms:Y()})]),dn=Ot([ut,Z({count:Y()})]),un=ut,cn=ut,pn=ut,fn=Ot([ut,Z({threshold:A($t(Y([Ce(0),we(1)])))})]),mn=Z({onAfter:A(hn),onClick:A(ln),onComplete:A(pn),onEnter:A(dn),onMouseEnter:A(un),onMouseLeave:A(cn),onShow:A(fn)}),vn=Ot([Mi,Z({playOnScroll:A(ii([Y([Ce(0),we(1)]),Y([Ce(0),we(1)])])),segments:A(be([ii([Y(),Y()]),z()]))})]);Ot([mn,Z({animationId:A(z()),playbackSettings:vn})]);var _n={jpeg:"image/jpeg",png:"image/png",gif:"image/gif",bmp:"image/bmp",svg:"image/svg+xml",webp:"image/webp",mpeg:"audio/mpeg",mp3:"audio/mp3"},oi={jpeg:[255,216,255],png:[137,80,78,71,13,10,26,10],gif:[71,73,70],bmp:[66,77],webp:[82,73,70,70,87,69,66,80],svg:[60,63,120],mp3:[73,68,51,3,0,0,0,0],mpeg:[73,68,51,3,0,0,0,0]},yn=t=>{let e=null,i=[];if(!t)return null;let o=t.substring(t.indexOf(",")+1);typeof window>"u"?e=Buffer.from(o,"base64").toString("binary"):e=atob(o);let n=new Uint8Array(e.length);for(let s=0;s<e.length;s+=1)n[s]=e.charCodeAt(s);i=Array.from(n.subarray(0,8));for(let s in oi){let a=oi[s];if(a&&i.every((h,r)=>h===a[r]))return _n[s]}return null},Te=class extends Error{constructor(t,e){super(t),Go(this,"code"),this.name="[dotlottie-js]",this.code=e}};function Pi(t){let e;if(typeof window>"u")e=Buffer.from(t).toString("base64");else{let i=Array.prototype.map.call(t,o=>String.fromCharCode(o)).join("");e=window.btoa(i)}return`data:${yn(e)};base64,${e}`}function ni(t){return"w"in t&&"h"in t&&!("xt"in t)&&"p"in t}function Le(t){return!("h"in t)&&!("w"in t)&&"p"in t&&"e"in t&&"u"in t&&"id"in t}async function Pt(t,e=()=>!0){if(!(t instanceof Uint8Array))throw new Te("DotLottie not found","INVALID_DOTLOTTIE");return await new Promise((i,o)=>{Mo(t,{filter:e},(n,s)=>{n&&o(n),i(s)})})}async function Ue(t,e,i){if(!(t instanceof Uint8Array))throw new Te("DotLottie not found","INVALID_DOTLOTTIE");return(await Pt(t,o=>o.name===e&&(!i||i(o))))[e]}async function Se(t){let e="manifest.json",i=(await Pt(t,o=>o.name===e))[e];if(!(typeof i>"u"))return JSON.parse(Ct(i,!1))}async function gn(t){if(!(t instanceof Uint8Array))return{success:!1,error:"DotLottie not found"};let e=await Se(t);if(typeof e>"u")return{success:!1,error:"Invalid .lottie file, manifest.json is missing"};let i=qo(an,e);return i.success?{success:!0}:{success:!1,error:`Invalid .lottie file, manifest.json structure is invalid, ${JSON.stringify(Po(i.error).nested,null,2)}`}}async function ri(t){let e=new Uint8Array(t),i=await gn(e);if(i.error)throw new Te(i.error,"INVALID_DOTLOTTIE");return e}async function bn(t,e){let i=await Pt(t,n=>{let s=n.name.replace("audio/","");return n.name.startsWith("audio/")&&(!e||e({...n,name:s}))}),o={};for(let n in i){let s=i[n];if(s instanceof Uint8Array){let a=n.replace("audio/","");o[a]=Pi(s)}}return o}async function wn(t,e){var n;let i=new Map;for(let[s,a]of Object.entries(e))for(let h of a.assets||[])if(Le(h)){let r=h.p;i.has(r)||i.set(r,new Set),(n=i.get(r))==null||n.add(s)}let o=await bn(t,s=>i.has(s.name));for(let[s,a]of i){let h=o[s];if(h)for(let r of a){let l=e[r];for(let d of(l==null?void 0:l.assets)||[])Le(d)&&d.p===s&&(d.p=h,d.u="",d.e=1)}}}async function Cn(t,e){let i=await Pt(t,n=>{let s=n.name.replace("images/","");return n.name.startsWith("images/")&&(!e||e({...n,name:s}))}),o={};for(let n in i){let s=i[n];if(s instanceof Uint8Array){let a=n.replace("images/","");o[a]=Pi(s)}}return o}async function Ln(t,e){var n;let i=new Map;for(let[s,a]of Object.entries(e))for(let h of a.assets||[])if(ni(h)){let r=h.p;i.has(r)||i.set(r,new Set),(n=i.get(r))==null||n.add(s)}let o=await Cn(t,s=>i.has(s.name));for(let[s,a]of i){let h=o[s];if(h)for(let r of a){let l=e[r];for(let d of(l==null?void 0:l.assets)||[])ni(d)&&d.p===s&&(d.p=h,d.u="",d.e=1)}}}async function Sn(t,e,{inlineAssets:i}={},o){let n=`animations/${e}.json`,s=await Ue(t,n,o);if(typeof s>"u")return;let a=JSON.parse(Ct(s,!1));if(!i)return a;let h={[e]:a};return await Ln(t,h),await wn(t,h),a}async function An(t,e,i){let o=`themes/${e}.lss`,n=await Ue(t,o,i);if(!(typeof n>"u"))return Ct(n,!1)}async function $n(t,e){let i={},o=await Pt(t,n=>{let s=n.name.replace("states/","").replace(".json","");return n.name.startsWith("states/")&&(!e||e({...n,name:s}))});for(let n in o){let s=o[n];if(s instanceof Uint8Array){let a=n.replace("states/","").replace(".json","");i[a]=Ct(s,!1)}}return i}async function kn(t,e,i){let o=`states/${e}.json`,n=await Ue(t,o,i);return typeof n>"u"?void 0:JSON.parse(Ct(n,!1))}Xo(nn());function C(t,e="dotLottie-common"){return new Error(`[${e}]: ${t}`)}function ft(t,e="dotLottie-common",...i){console.error(`[${e}]:`,t,...i)}function x(t,e="dotLottie-common",...i){console.warn(`[${e}]:`,t,...i)}function xn(t=""){let e=t.trim(),i=e.lastIndexOf("/"),o=e.substring(i+1),n=o.indexOf(".");return n!==-1?o.substring(0,n):o}function qt(t){return["v","ip","op","layers","fr","w","h"].every(e=>Object.prototype.hasOwnProperty.call(t,e))}function En(t){let e=t.assets;return e?e.some(i=>Le(i)):!1}function In(t){try{let e=JSON.parse(t);return qt(e)}catch{return!1}}function ar(t,e){let i=Object.keys(t).find(o=>t[o]===e);if(i===void 0)throw new Error("Value not found in the object.");return i}var On=class{constructor(){p(this,"_dotLottie");p(this,"_animationsMap",new Map);p(this,"_themeMap",new Map);p(this,"_stateMachinesMap",new Map);p(this,"_manifest")}get dotLottie(){return this._dotLottie}get animationsMap(){return this._animationsMap}get themeMap(){return this._themeMap}get stateMachinesMap(){return this._stateMachinesMap}get manifest(){return this._manifest}async loadFromUrl(t){var i;let e=await fetch(t,{method:"GET",mode:"cors"});if(!e.ok)throw new Error(`Failed to load dotLottie from ${t} with status ${e.status}`);if((i=e.headers.get("content-type"))!=null&&i.includes("application/json")){let o=await e.json();if(!qt(o))throw new Error(`Invalid lottie JSON at ${t}`);let n=xn(t);this._animationsMap.set(n,o);let s={activeAnimationId:n,animations:[{id:n}]};this._manifest=s}else{this._dotLottie=await ri(await e.arrayBuffer());let o=await Se(this._dotLottie);if(!o)throw new Error("Manifest not found");this._manifest=o}}loadFromLottieJSON(t){if(!qt(t))throw new Error("Invalid lottie JSON");let e="my-animation";this._animationsMap.set(e,t);let i={activeAnimationId:e,animations:[{id:e}]};this._manifest=i}async loadFromArrayBuffer(t){this._dotLottie=await ri(t);let e=await Se(this._dotLottie);if(!e)throw new Error("Manifest not found");this._manifest=e}async getAnimation(t){if(this._animationsMap.get(t))return this._animationsMap.get(t);if(!this._dotLottie)return;let e=await Sn(this._dotLottie,t,{inlineAssets:!0});return e&&this._animationsMap.set(t,e),e}async getTheme(t){if(this._themeMap.get(t))return this._themeMap.get(t);if(!this._dotLottie)return;let e=await An(this._dotLottie,t);return e&&this._themeMap.set(t,e),e}async getStateMachines(){if(!this._dotLottie)return;let t=await $n(this._dotLottie);for(let e in t)if(e){let i=t[e];if(i){let o=JSON.parse(i);if(o){let n=o.descriptor.id;this._stateMachinesMap.get(n)||this._stateMachinesMap.set(n,o)}}}return Array.from(this._stateMachinesMap.values())}async getStateMachine(t){if(this._stateMachinesMap.get(t))return this._stateMachinesMap.get(t);if(!this._dotLottie)return;let e=await kn(this._dotLottie,t);return e&&this._stateMachinesMap.set(e.descriptor.id,e),e}};async function Mn(t,e){let[{relottie:i},{default:o}]=await Promise.all([W(()=>import("./dist-AIQGIF54-ZRQFFVOU.bfc94ab8.js"),["./dist-AIQGIF54-ZRQFFVOU.bfc94ab8.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url),W(()=>import("./dist-4JHQ5UB4-ADENUQD3.af61c3e3.js"),["./dist-4JHQ5UB4-ADENUQD3.af61c3e3.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url)]),n=await i().use(o,{lss:e}).process(JSON.stringify(t));return JSON.parse(n.value)}function Kt(){throw new Error("Cycle detected")}function je(){if(yt>1)yt--;else{for(var t,e=!1;kt!==void 0;){var i=kt;for(kt=void 0,Ae++;i!==void 0;){var o=i.o;if(i.o=void 0,i.f&=-3,!(8&i.f)&&Ui(i))try{i.c()}catch(n){e||(t=n,e=!0)}i=o}}if(Ae=0,yt--,e)throw t}}var E=void 0,kt=void 0,yt=0,Ae=0,Rt=0;function Ti(t){if(E!==void 0){var e=t.n;if(e===void 0||e.t!==E)return e={i:0,S:t,p:E.s,n:void 0,t:E,e:void 0,x:void 0,r:e},E.s!==void 0&&(E.s.n=e),E.s=e,t.n=e,32&E.f&&t.S(e),e;if(e.i===-1)return e.i=0,e.n!==void 0&&(e.n.p=e.p,e.p!==void 0&&(e.p.n=e.n),e.p=E.s,e.n=void 0,E.s.n=e,E.s=e),e}}function H(t){this.v=t,this.i=0,this.n=void 0,this.t=void 0}H.prototype.h=function(){return!0};H.prototype.S=function(t){this.t!==t&&t.e===void 0&&(t.x=this.t,this.t!==void 0&&(this.t.e=t),this.t=t)};H.prototype.U=function(t){if(this.t!==void 0){var e=t.e,i=t.x;e!==void 0&&(e.x=i,t.e=void 0),i!==void 0&&(i.e=e,t.x=void 0),t===this.t&&(this.t=i)}};H.prototype.subscribe=function(t){var e=this;return Un(function(){var i=e.value,o=32&this.f;this.f&=-33;try{t(i)}finally{this.f|=o}})};H.prototype.valueOf=function(){return this.value};H.prototype.toString=function(){return this.value+""};H.prototype.toJSON=function(){return this.value};H.prototype.peek=function(){return this.v};Object.defineProperty(H.prototype,"value",{get:function(){var t=Ti(this);return t!==void 0&&(t.i=this.i),this.v},set:function(t){if(E instanceof ct&&function(){throw new Error("Computed cannot have side-effects")}(),t!==this.v){Ae>100&&Kt(),this.v=t,this.i++,Rt++,yt++;try{for(var e=this.t;e!==void 0;e=e.x)e.t.N()}finally{je()}}}});function Pn(t){return new H(t)}function Ui(t){for(var e=t.s;e!==void 0;e=e.n)if(e.S.i!==e.i||!e.S.h()||e.S.i!==e.i)return!0;return!1}function ji(t){for(var e=t.s;e!==void 0;e=e.n){var i=e.S.n;if(i!==void 0&&(e.r=i),e.S.n=e,e.i=-1,e.n===void 0){t.s=e;break}}}function Ni(t){for(var e=t.s,i=void 0;e!==void 0;){var o=e.p;e.i===-1?(e.S.U(e),o!==void 0&&(o.n=e.n),e.n!==void 0&&(e.n.p=o)):i=e,e.S.n=e.r,e.r!==void 0&&(e.r=void 0),e=o}t.s=i}function ct(t){H.call(this,void 0),this.x=t,this.s=void 0,this.g=Rt-1,this.f=4}(ct.prototype=new H).h=function(){if(this.f&=-3,1&this.f)return!1;if((36&this.f)==32||(this.f&=-5,this.g===Rt))return!0;if(this.g=Rt,this.f|=1,this.i>0&&!Ui(this))return this.f&=-2,!0;var t=E;try{ji(this),E=this;var e=this.x();(16&this.f||this.v!==e||this.i===0)&&(this.v=e,this.f&=-17,this.i++)}catch(i){this.v=i,this.f|=16,this.i++}return E=t,Ni(this),this.f&=-2,!0};ct.prototype.S=function(t){if(this.t===void 0){this.f|=36;for(var e=this.s;e!==void 0;e=e.n)e.S.S(e)}H.prototype.S.call(this,t)};ct.prototype.U=function(t){if(this.t!==void 0&&(H.prototype.U.call(this,t),this.t===void 0)){this.f&=-33;for(var e=this.s;e!==void 0;e=e.n)e.S.U(e)}};ct.prototype.N=function(){if(!(2&this.f)){this.f|=6;for(var t=this.t;t!==void 0;t=t.x)t.t.N()}};ct.prototype.peek=function(){if(this.h()||Kt(),16&this.f)throw this.v;return this.v};Object.defineProperty(ct.prototype,"value",{get:function(){1&this.f&&Kt();var t=Ti(this);if(this.h(),t!==void 0&&(t.i=this.i),16&this.f)throw this.v;return this.v}});function zi(t){var e=t.u;if(t.u=void 0,typeof e=="function"){yt++;var i=E;E=void 0;try{e()}catch(o){throw t.f&=-2,t.f|=8,Ne(t),o}finally{E=i,je()}}}function Ne(t){for(var e=t.s;e!==void 0;e=e.n)e.S.U(e);t.x=void 0,t.s=void 0,zi(t)}function Tn(t){if(E!==this)throw new Error("Out-of-order effect");Ni(this),E=t,this.f&=-2,8&this.f&&Ne(this),je()}function Tt(t){this.x=t,this.u=void 0,this.s=void 0,this.o=void 0,this.f=32}Tt.prototype.c=function(){var t=this.S();try{if(8&this.f||this.x===void 0)return;var e=this.x();typeof e=="function"&&(this.u=e)}finally{t()}};Tt.prototype.S=function(){1&this.f&&Kt(),this.f|=1,this.f&=-9,zi(this),ji(this),yt++;var t=E;return E=this,Tn.bind(this,t)};Tt.prototype.N=function(){2&this.f||(this.f|=2,this.o=kt,kt=this)};Tt.prototype.d=function(){this.f|=8,1&this.f||Ne(this)};function Un(t){var e=new Tt(t);try{e.c()}catch(i){throw e.d(),i}return e.d.bind(e)}var jn=class{constructor(t){p(this,"_state");p(this,"_prevState");this._prevState=t,this._state=Pn(t)}setState(t){this._prevState=this._state.value,this._state.value=t}subscribe(t){return this._state.subscribe(e=>t(e,this._prevState))}};async function Nn(t,e){let[{DotLottieStateMachineManager:i}]=await Promise.all([W(()=>import("./dotlottie-state-machine-manager-EQLJDSHN-GQNUJPNH.e482c915.js"),["./dotlottie-state-machine-manager-EQLJDSHN-GQNUJPNH.e482c915.js","./entry.2f09776e.js","./entry.b0b29391.css"],import.meta.url)]);if(!t.length)throw C("No state machines available inside this .lottie!");return new i(t,e)}var zn={name:"@dotlottie/common",version:"0.7.0",type:"module",description:"",author:"Afsal <afsal@lottiefiles.com>, Sam Osborne <sam@lottiefiles.com>",license:"MIT",private:!0,engines:{node:">18.0.0"},module:"dist/index.js",types:"dist/index.d.ts",files:["dist"],keywords:[],scripts:{build:"tsup",dev:"tsup --watch",lint:"eslint .","type-check":"tsc --noEmit"},dependencies:{"@dotlottie/dotlottie-js":"0.6.0","@lottiefiles/relottie":"1.0.0","@lottiefiles/relottie-style":"0.4.1","@preact/signals-core":"^1.2.3",howler:"^2.2.3","lottie-web":"^5.12.2",xstate:"^4.38.1"},devDependencies:{"@lottiefiles/lottie-types":"^1.2.0","@types/howler":"^2.2.8",tsup:"^6.1.3",typescript:"^4.7.4"},publishConfig:{access:"restricted"}},ot=(t=>(t.Complete="complete",t.DataFail="data_fail",t.DataReady="data_ready",t.Error="error",t.Frame="frame",t.Freeze="freeze",t.LoopComplete="loopComplete",t.Pause="pause",t.Play="play",t.Ready="ready",t.Stop="stop",t.VisibilityChange="visibilityChange",t))(ot||{}),vt=(t=>(t.Completed="completed",t.Error="error",t.Fetching="fetching",t.Frozen="frozen",t.Initial="initial",t.Loading="loading",t.Paused="paused",t.Playing="playing",t.Ready="ready",t.Stopped="stopped",t))(vt||{}),Di=(t=>(t.Bounce="bounce",t.Normal="normal",t))(Di||{}),q={autoplay:!1,direction:1,hover:!1,intermission:0,loop:!1,playMode:"normal",speed:1,defaultTheme:""},Vi={activeStateId:"",autoplay:!1,currentState:"initial",frame:0,seeker:0,direction:1,hover:!1,loop:!1,playMode:"normal",speed:1,background:"transparent",intermission:0,currentAnimationId:void 0,visibilityPercentage:0},si=class{constructor(t,e,i){p(this,"_lottie");p(this,"_src");p(this,"_animationConfig");p(this,"_prevUserPlaybackOptions",{});p(this,"_userPlaybackOptions");p(this,"_hover",!1);p(this,"_loop",!1);p(this,"_counter",0);p(this,"_intermission",0);p(this,"_counterInterval",null);p(this,"_container",null);p(this,"_name");p(this,"_mode","normal");p(this,"_background","transparent");p(this,"_animation");p(this,"_defaultTheme");p(this,"_activeAnimationId");p(this,"_currentAnimationId");p(this,"_testId");p(this,"_listeners",new Map);p(this,"_currentState","initial");p(this,"_stateBeforeFreeze","initial");p(this,"state",new jn(Vi));p(this,"_light",!1);p(this,"_worker",!1);p(this,"_dotLottieLoader",new On);p(this,"_activeStateId");p(this,"_inInteractiveMode",!1);p(this,"_scrollTicking",!1);p(this,"_scrollCallback");p(this,"_onShowIntersectionObserver");p(this,"_visibilityPercentage",0);p(this,"_audios",[]);p(this,"_stateMachineManager");this._src=structuredClone(t),i!=null&&i.testId&&(this._testId=i.testId),this._defaultTheme=(i==null?void 0:i.defaultTheme)||"",this._userPlaybackOptions=this._validatePlaybackOptions(i||{}),typeof(i==null?void 0:i.activeAnimationId)=="string"&&(this._activeAnimationId=i.activeAnimationId),this._container=e||null,typeof(i==null?void 0:i.background)=="string"&&this.setBackground(i.background),typeof(i==null?void 0:i.activeStateId)<"u"&&(this._activeStateId=i.activeStateId);let{rendererSettings:o,...n}=i||{};this._animationConfig={loop:!1,autoplay:!1,renderer:"svg",rendererSettings:{clearCanvas:!0,progressiveLoad:!0,hideOnTransparent:!0,filterSize:{width:"200%",height:"200%",x:"-50%",y:"-50%"},...o},...n},i!=null&&i.light&&(this._light=i.light),i!=null&&i.worker&&(this._worker=i.worker),this._listenToHover(),this._listenToVisibilityChange()}_listenToHover(){var i,o,n,s;let t=()=>{this._hover&&this.currentState!=="playing"&&this.play()},e=()=>{this._hover&&this.currentState==="playing"&&this.stop()};(i=this._container)==null||i.removeEventListener("mouseenter",t),(o=this._container)==null||o.removeEventListener("mouseleave",e),(n=this._container)==null||n.addEventListener("mouseleave",e),(s=this._container)==null||s.addEventListener("mouseenter",t)}_onVisibilityChange(){!this._lottie||typeof document>"u"||(document.hidden&&this.currentState==="playing"?this.freeze():this.currentState==="frozen"&&this.unfreeze())}_listenToVisibilityChange(){typeof document<"u"&&typeof document.hidden<"u"&&document.addEventListener("visibilitychange",()=>this._onVisibilityChange())}_getOption(t){var i;if(typeof this._userPlaybackOptions[t]<"u")return this._userPlaybackOptions[t];let e=(i=this._dotLottieLoader.manifest)==null?void 0:i.animations.find(o=>o.id===this._currentAnimationId);return e&&typeof e[t]<"u"?e[t]:q[t]}_getPlaybackOptions(){let t={};for(let e in q)typeof q[e]<"u"&&(t[e]=this._getOption(e));return t}_setPlayerState(t){var i,o,n;let e=t(this._getPlaybackOptions());try{Mi._parse(e)}catch{x(`Invalid PlaybackOptions, ${JSON.stringify(e,null,2)}`);return}typeof e.defaultTheme<"u"&&(this._defaultTheme=e.defaultTheme),typeof e.playMode<"u"&&(this._mode=e.playMode),typeof e.intermission<"u"&&(this._intermission=e.intermission),typeof e.hover<"u"&&(this._hover=e.hover),typeof e.loop<"u"&&(this.clearCountTimer(),this._loop=e.loop,this._counter=0,(i=this._lottie)==null||i.setLoop(typeof e.loop=="number"?!0:e.loop)),typeof e.speed<"u"&&((o=this._lottie)==null||o.setSpeed(e.speed)),typeof e.autoplay<"u"&&this._lottie&&(this._lottie.autoplay=e.autoplay),typeof e.direction<"u"&&((n=this._lottie)==null||n.setDirection(e.direction))}_getOptionsFromAnimation(t){let{id:e,...i}=t;return{...q,...i}}_updateTestData(){!this._testId||!this._lottie||(window.dotLottiePlayer||(window.dotLottiePlayer={[this._testId]:{}}),window.dotLottiePlayer[this._testId]={direction:this._lottie.playDirection,currentState:this._currentState,loop:this.loop,mode:this._mode,speed:this._lottie.playSpeed})}setContainer(t){t!==this._container&&(this._container=t,this.setBackground(this._background),this._listenToHover())}get currentState(){return this._currentState}clearCountTimer(){this._counterInterval&&clearInterval(this._counterInterval)}setCurrentState(t){this._currentState=t,this._notify(),this._updateTestData()}static isPathJSON(t){var e;return((e=t.split(".").pop())==null?void 0:e.toLowerCase())==="json"}get src(){return this._src}updateSrc(t){this._src!==t&&(this._src=structuredClone(t),this._activeAnimationId=void 0,this._currentAnimationId=void 0,this.load())}get intermission(){return this._intermission}get hover(){return this._hover}setHover(t){typeof t=="boolean"&&(this._hover=t,this._userPlaybackOptions.hover=t,this._notify())}setIntermission(t){this._intermission=t,this._userPlaybackOptions.intermission=t,this._notify()}get mode(){return this._mode}get animations(){return this._dotLottieLoader.animationsMap}get themes(){return this._dotLottieLoader.themeMap}setMode(t){typeof t=="string"&&(this._mode=t,this._userPlaybackOptions.playMode=t,this._setPlayerState(()=>({playMode:t})),this._notify(),this._updateTestData())}get container(){if(this._container)return this._container}goToAndPlay(t,e,i){if(!this._lottie||["loading"].includes(this._currentState)){x("goToAndPlay() Can't use whilst loading.");return}this._lottie.goToAndPlay(t,e,i),this.setCurrentState("playing")}goToAndStop(t,e,i){if(!this._lottie||["loading"].includes(this._currentState)){x("goToAndStop() Can't use whilst loading.");return}this._lottie.goToAndStop(t,e,i),this.setCurrentState("stopped")}seek(t){if(!this._lottie||["loading"].includes(this._currentState)){x("seek() Can't use whilst loading.");return}let e=t;typeof e=="number"&&(e=Math.round(e));let i=/^(\d+)(%?)$/u.exec(e.toString());if(!i)return;let o=i[2]==="%"?this.totalFrames*Number(i[1])/100:i[1];o!==void 0&&(this._lottie.goToAndPlay(o,!0),this.currentState==="playing"?this.play():this.currentState==="frozen"?this.freeze():this.pause())}_areNumbersInRange(t,e){return t>=0&&t<=1&&e>=0&&e<=1}_updatePosition(t,e,i){let[o,n]=t??[0,this.totalFrames-1],[s,a]=e??[0,1];if(!this._areNumbersInRange(s,a)){ft("threshold values must be between 0 and 1");return}if(this.container){let{height:h,top:r}=this.container.getBoundingClientRect(),l=window.innerHeight-r,d=window.innerHeight+h,u=l/d,c=o+Math.round((u-s)/(a-s)*(n-o));i&&i(u),this.goToAndStop(c,!0),(c>=n||u>=a)&&this._handleAnimationComplete()}this._scrollTicking=!1}_requestTick(t,e,i){this._scrollTicking||(requestAnimationFrame(()=>this._updatePosition(t,e,i)),this._scrollTicking=!0)}playOnScroll(t){this.stop(),this._scrollCallback&&this.stopPlayOnScroll(),this._scrollCallback=()=>this._requestTick(t==null?void 0:t.segments,t==null?void 0:t.threshold,t==null?void 0:t.positionCallback),window.addEventListener("scroll",this._scrollCallback)}stopPlayOnScroll(){this._scrollCallback&&(window.removeEventListener("scroll",this._scrollCallback),this._scrollCallback=void 0)}stopPlayOnShow(){this._onShowIntersectionObserver&&(this._onShowIntersectionObserver.disconnect(),this._onShowIntersectionObserver=void 0)}addIntersectionObserver(t){if(!this.container)throw C("Can't play on show, player container element not available.");let e={root:null,rootMargin:"0px",threshold:t!=null&&t.threshold?t.threshold:[0,1]},i=o=>{o.forEach(n=>{var s,a;this._visibilityPercentage=n.intersectionRatio*100,n.isIntersecting?(t!=null&&t.callbackOnIntersect&&t.callbackOnIntersect(this._visibilityPercentage),(s=this._container)==null||s.dispatchEvent(new Event("visibilityChange"))):t!=null&&t.callbackOnIntersect&&(t.callbackOnIntersect(0),(a=this._container)==null||a.dispatchEvent(new Event("visibilityChange")))})};this._onShowIntersectionObserver=new IntersectionObserver(i,e),this._onShowIntersectionObserver.observe(this.container)}playOnShow(t){if(this.stop(),!this.container)throw C("Can't play on show, player container element not available.");this._onShowIntersectionObserver&&this.stopPlayOnShow(),this.addIntersectionObserver({threshold:(t==null?void 0:t.threshold)??[],callbackOnIntersect:e=>{e===0?this.pause():this.play()}})}_validatePlaybackOptions(t){if(!t)return{};let e={};for(let[i,o]of Object.entries(t))switch(i){case"autoplay":typeof o=="boolean"&&(e.autoplay=o);break;case"direction":typeof o=="number"&&[1,-1].includes(o)&&(e.direction=o);break;case"loop":(typeof o=="boolean"||typeof o=="number")&&(e.loop=o);break;case"playMode":typeof o=="string"&&["normal","bounce"].includes(o)&&(e.playMode=o);break;case"speed":typeof o=="number"&&(e.speed=o);break;case"themeColor":typeof o=="string"&&(e.themeColor=o);break;case"hover":typeof o=="boolean"&&(e.hover=o);break;case"intermission":typeof o=="number"&&(e.intermission=o);break;case"defaultTheme":typeof o=="string"&&(e.defaultTheme=o);break}return this._requireValidPlaybackOptions(e),e}_requireAnimationsInTheManifest(){var t;if(!((t=this._dotLottieLoader.manifest)!=null&&t.animations.length))throw C("No animations found in manifest.")}_requireAnimationsToBeLoaded(){if(this._dotLottieLoader.animationsMap.size===0)throw C("No animations have been loaded.")}async play(t,e){var i,o;if(["initial","loading"].includes(this._currentState)){x("Player unable to play whilst loading.");return}if(this._requireAnimationsInTheManifest(),this._requireAnimationsToBeLoaded(),this._lottie&&!t){this._lottie.playDirection===-1&&this._lottie.currentFrame===0?this._lottie.goToAndPlay(this._lottie.totalFrames,!0):this._lottie.play(),this.setCurrentState("playing");return}if(typeof t=="number"){let n=(i=this._dotLottieLoader.manifest)==null?void 0:i.animations[t];if(!n)throw C("animation not found.");typeof e=="function"?await this.render({id:n.id,...e(this._getPlaybackOptions(),this._getOptionsFromAnimation(n))}):await this.render({id:n.id})}if(typeof t=="string"){let n=(o=this._dotLottieLoader.manifest)==null?void 0:o.animations.find(s=>s.id===t);if(!n)throw C("animation not found.");typeof e=="function"?await this.render({id:n.id,...e(this._getPlaybackOptions(),this._getOptionsFromAnimation(n))}):await this.render({id:n.id})}}playSegments(t,e){if(!this._lottie||["loading"].includes(this._currentState)){x("playSegments() Can't use whilst loading.");return}this._lottie.playSegments(t,e),this.setCurrentState("playing")}resetSegments(t){if(!this._lottie||["loading"].includes(this._currentState)){x("resetSegments() Can't use whilst loading.");return}this._lottie.resetSegments(t)}togglePlay(){this.currentState==="playing"?this.pause():this.play()}_getAnimationByIdOrIndex(t){var e,i;if(this._requireAnimationsInTheManifest(),this._requireAnimationsToBeLoaded(),typeof t=="number"){let o=(e=this._dotLottieLoader.manifest)==null?void 0:e.animations[t];if(!o)throw C("animation not found.");return o}if(typeof t=="string"){let o=(i=this._dotLottieLoader.manifest)==null?void 0:i.animations.find(n=>n.id===t);if(!o)throw C("animation not found.");return o}throw C("first param must be a number or string")}get activeAnimationId(){return this._getActiveAnimationId()}get currentAnimationId(){return this._currentAnimationId}get activeStateId(){return this._activeStateId}async _startInteractivity(t){if(!this._inInteractiveMode){ft("Can't start interactivity. Not in interactive mode. Call enterInteractiveMode(stateId: string) to start.");return}if(this._dotLottieLoader.stateMachinesMap.size===0&&await this._dotLottieLoader.getStateMachines(),this._dotLottieLoader.stateMachinesMap.size===0)throw C("No interactivity states are available.");if(t==="undefined")throw C("stateId is not specified.");this._stateMachineManager||(this._stateMachineManager=await Nn(Array.from(this._dotLottieLoader.stateMachinesMap.values()),this)),this._stateMachineManager.start(t)}enterInteractiveMode(t){var e;if(t)this._inInteractiveMode||(this._prevUserPlaybackOptions={...this._userPlaybackOptions}),this._inInteractiveMode&&((e=this._stateMachineManager)==null||e.stop()),this._activeStateId=t,this._inInteractiveMode=!0,this._startInteractivity(t);else throw C("stateId must be a non-empty string.")}exitInteractiveMode(){var t;this._inInteractiveMode&&(this._inInteractiveMode=!1,this._activeStateId="",(t=this._stateMachineManager)==null||t.stop(),this._userPlaybackOptions={},this._userPlaybackOptions={...this._prevUserPlaybackOptions},this._prevUserPlaybackOptions={},this.reset())}reset(){var i;let t=this._getActiveAnimationId(),e=(i=this._dotLottieLoader.manifest)==null?void 0:i.animations.find(o=>o.id===t);if(this._inInteractiveMode&&this.exitInteractiveMode(),!e)throw C("animation not found.");this.play(t)}previous(t){if(!this._dotLottieLoader.manifest||!this._dotLottieLoader.manifest.animations.length)throw C("manifest not found.");if(this._inInteractiveMode){x("previous() is not supported in interactive mode.");return}let e=this._dotLottieLoader.manifest.animations.findIndex(o=>o.id===this._currentAnimationId);if(e===-1)throw C("animation not found.");let i=this._dotLottieLoader.manifest.animations[(e-1+this._dotLottieLoader.manifest.animations.length)%this._dotLottieLoader.manifest.animations.length];if(!i||!i.id)throw C("animation not found.");typeof t=="function"?this.render({id:i.id,...t(this._getPlaybackOptions(),this._getOptionsFromAnimation(i))}):this.render({id:i.id})}next(t){if(!this._dotLottieLoader.manifest||!this._dotLottieLoader.manifest.animations.length)throw C("manifest not found.");if(this._inInteractiveMode){x("next() is not supported in interactive mode.");return}let e=this._dotLottieLoader.manifest.animations.findIndex(o=>o.id===this._currentAnimationId);if(e===-1)throw C("animation not found.");let i=this._dotLottieLoader.manifest.animations[(e+1)%this._dotLottieLoader.manifest.animations.length];if(!i||!i.id)throw C("animation not found.");typeof t=="function"?this.render({id:i.id,...t(this._getPlaybackOptions(),this._getOptionsFromAnimation(i))}):this.render({id:i.id})}getManifest(){return this._dotLottieLoader.manifest}resize(){if(!this._lottie||["loading"].includes(this._currentState)){x("resize() Can't use whilst loading.");return}this._lottie.resize()}stop(){if(!this._lottie||["loading"].includes(this._currentState)){x("stop() Can't use whilst loading.");return}this.clearCountTimer(),this._counter=0,this._setPlayerState(()=>({direction:this._getOption("direction")})),this._lottie.stop(),this.setCurrentState("stopped")}pause(){if(!this._lottie||["loading"].includes(this._currentState)){x("pause() Can't use whilst loading.");return}this.clearCountTimer(),this._lottie.pause(),this.setCurrentState("paused")}freeze(){if(!this._lottie||["loading"].includes(this._currentState)){x("freeze() Can't use whilst loading.");return}this.currentState!=="frozen"&&(this._stateBeforeFreeze=this.currentState),this._lottie.pause(),this.setCurrentState("frozen")}unfreeze(){if(!this._lottie||["loading"].includes(this._currentState)){x("unfreeze() Can't use whilst loading.");return}this._stateBeforeFreeze==="playing"?this.play():this.pause()}destroy(){var t,e;(t=this._container)!=null&&t.__lottie&&(this._container.__lottie.destroy(),this._container.__lottie=null),this._audios.length&&(this._audios.forEach(i=>{i.unload()}),this._audios=[]),this.clearCountTimer(),typeof document<"u"&&document.removeEventListener("visibilitychange",()=>this._onVisibilityChange()),this._counter=0,(e=this._lottie)==null||e.destroy()}getAnimationInstance(){return this._lottie}static getLottieWebVersion(){return`${zn.dependencies["lottie-web"]}`}addEventListener(t,e){var i,o,n;this._listeners.has(t)||this._listeners.set(t,new Set),(i=this._listeners.get(t))==null||i.add(e);try{t==="complete"?(o=this._container)==null||o.addEventListener(t,e):(n=this._lottie)==null||n.addEventListener(t,e)}catch(s){ft(`addEventListener ${s}`)}}getState(){var t,e,i;return{autoplay:((t=this._lottie)==null?void 0:t.autoplay)??!1,currentState:this._currentState,frame:this._frame,visibilityPercentage:this._visibilityPercentage,seeker:this._seeker,direction:((e=this._lottie)==null?void 0:e.playDirection)??1,hover:this._hover,loop:this._loop||!1,playMode:this._mode,speed:((i=this._lottie)==null?void 0:i.playSpeed)??1,background:this._background,intermission:this._intermission,defaultTheme:this._defaultTheme,currentAnimationId:this._currentAnimationId,activeStateId:this._activeStateId??""}}_notify(){this.state.setState(this.getState())}get totalFrames(){var t;return((t=this._lottie)==null?void 0:t.totalFrames)||0}get direction(){return this._lottie?this._lottie.playDirection:1}setDirection(t){this._requireValidDirection(t),this._setPlayerState(()=>({direction:t})),this._userPlaybackOptions.direction=t}get speed(){var t;return((t=this._lottie)==null?void 0:t.playSpeed)||1}setSpeed(t){this._requireValidSpeed(t),this._setPlayerState(()=>({speed:t})),this._userPlaybackOptions.speed=t}get autoplay(){var t;return((t=this._lottie)==null?void 0:t.autoplay)??!1}setAutoplay(t){if(this._requireValidAutoplay(t),!this._lottie||["loading"].includes(this._currentState)){x("setAutoplay() Can't use whilst loading.");return}this._setPlayerState(()=>({autoplay:t})),this._userPlaybackOptions.autoplay=t}toggleAutoplay(){if(!this._lottie||["loading"].includes(this._currentState)){x("toggleAutoplay() Can't use whilst loading.");return}this.setAutoplay(!this._lottie.autoplay)}get defaultTheme(){return this._defaultTheme}setDefaultTheme(t){this._setPlayerState(()=>({defaultTheme:t})),this._userPlaybackOptions.defaultTheme=t,this._animation&&this.render()}get loop(){return this._loop}setLoop(t){this._requireValidLoop(t),this._setPlayerState(()=>({loop:t})),this._userPlaybackOptions.loop=t}toggleLoop(){if(!this._lottie||["loading"].includes(this._currentState)){x("toggleLoop() Can't use whilst loading.");return}this.setLoop(!this._loop)}get background(){return this._background}setBackground(t){this._requireValidBackground(t),this._background=t,this._container&&(this._container.style.backgroundColor=t)}get _frame(){return this._lottie?this.currentState==="completed"?this.direction===-1?0:this._lottie.totalFrames:this._lottie.currentFrame:0}get _seeker(){return this._lottie?this._frame/this._lottie.totalFrames*100:0}async revertToManifestValues(t){var o;let e;!Array.isArray(t)||t.length===0?e=["autoplay","defaultTheme","direction","hover","intermission","loop","playMode","speed","activeAnimationId"]:e=t;let i=!1;if(e.includes("activeAnimationId")){let n=(o=this._dotLottieLoader.manifest)==null?void 0:o.activeAnimationId,s=this._getAnimationByIdOrIndex(n||0);this._activeAnimationId=n,await this._setCurrentAnimation(s.id),i=!0}e.forEach(n=>{switch(n){case"autoplay":delete this._userPlaybackOptions.autoplay,this.setAutoplay(this._getOption("autoplay"));break;case"defaultTheme":delete this._userPlaybackOptions.defaultTheme,this.setDefaultTheme(this._getOption("defaultTheme"));break;case"direction":delete this._userPlaybackOptions.direction,this.setDirection(this._getOption("direction"));break;case"hover":delete this._userPlaybackOptions.hover,this.setHover(this._getOption("hover"));break;case"intermission":delete this._userPlaybackOptions.intermission,this.setIntermission(this._getOption("intermission"));break;case"loop":delete this._userPlaybackOptions.loop,this.setLoop(this._getOption("loop"));break;case"playMode":delete this._userPlaybackOptions.playMode,this.setMode(this._getOption("playMode")),this.setDirection(this._getOption("direction"));break;case"speed":delete this._userPlaybackOptions.speed,this.setSpeed(this._getOption("speed"));break}}),i&&this.render()}removeEventListener(t,e){var i,o,n;try{t==="complete"?(i=this._container)==null||i.removeEventListener(t,e):(o=this._lottie)==null||o.removeEventListener(t,e),(n=this._listeners.get(t))==null||n.delete(e)}catch(s){ft("removeEventListener",s)}}_handleAnimationComplete(){var e;typeof this._loop=="number"&&this.stop();let t=this.direction===-1?0:this.totalFrames;this.goToAndStop(t,!0),this._counter=0,this.clearCountTimer(),this.setCurrentState("completed"),(e=this._container)==null||e.dispatchEvent(new Event("complete"))}addEventListeners(){var t;if(!this._lottie||["loading"].includes(this._currentState)){x("addEventListeners() Can't use whilst loading.");return}this._lottie.addEventListener("enterFrame",()=>{var e;if(!this._lottie){x("enterFrame event : Lottie is undefined.");return}Math.floor(this._lottie.currentFrame)===0&&this.direction===-1&&((e=this._container)==null||e.dispatchEvent(new Event("complete")),this.loop||this.setCurrentState("completed")),this._notify()}),this._lottie.addEventListener("loopComplete",()=>{var o;if(!this._lottie){x("loopComplete event : Lottie is undefined.");return}(o=this._container)==null||o.dispatchEvent(new Event("loopComplete")),this.intermission>0&&this.pause();let e=this._lottie.playDirection;if(typeof this._loop=="number"&&this._loop>0&&(this._counter+=this._mode==="bounce"?.5:1,this._counter>=this._loop)){this._handleAnimationComplete();return}this._mode==="bounce"&&typeof e=="number"&&(e=Number(e)*-1);let i=e===-1?this._lottie.totalFrames-1:0;this.intermission?(this.goToAndPlay(i,!0),this.pause(),this._counterInterval=window.setTimeout(()=>{this._lottie&&(this._setPlayerState(()=>({direction:e})),this.goToAndPlay(i,!0))},this._intermission)):(this._setPlayerState(()=>({direction:e})),this.goToAndPlay(e===-1?this.totalFrames-1:0,!0))}),this._lottie.addEventListener("complete",()=>{if(this._lottie&&this._loop===!1&&this._mode==="bounce"){if(this._counter+=.5,this._counter>=1){this._handleAnimationComplete();return}this._counterInterval=window.setTimeout(()=>{if(!this._lottie)return;let e=this._lottie.playDirection;this._mode==="bounce"&&typeof e=="number"&&(e=Number(e)*-1);let i=e===-1?this.totalFrames-1:0;this._setPlayerState(()=>({direction:e})),this.goToAndPlay(i,!0)},this._intermission)}else this._handleAnimationComplete()});for(let[e,i]of this._listeners)if(e==="complete")for(let o of i)(t=this._container)==null||t.addEventListener(e,o);else for(let o of i)this._lottie.addEventListener(e,o)}async _setCurrentAnimation(t){this._currentState="loading";let e=await this._dotLottieLoader.getAnimation(t);this._currentAnimationId=t,this._animation=e,this._currentState="ready"}async _getAudioFactory(){if(this._animation&&En(this._animation)){let{DotLottieAudio:t}=await W(()=>import("./dotlottie-audio-FLUSZPLH.2cd9ecde.js"),["./dotlottie-audio-FLUSZPLH.2cd9ecde.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url);return e=>{let i=new t({src:[e]});return this._audios.push(i),i}}return null}async render(t){if(t!=null&&t.id)await this._setCurrentAnimation(t.id);else if(!this._animation)throw C("no animation selected");let e=q.loop,i=q.autoplay,o=q.playMode,n=q.intermission,s=q.hover,a=q.direction,h=q.speed,r=q.defaultTheme;e=(t==null?void 0:t.loop)??this._getOption("loop"),i=(t==null?void 0:t.autoplay)??this._getOption("autoplay"),o=(t==null?void 0:t.playMode)??this._getOption("playMode"),n=(t==null?void 0:t.intermission)??this._getOption("intermission"),s=(t==null?void 0:t.hover)??this._getOption("hover"),a=(t==null?void 0:t.direction)??this._getOption("direction"),h=(t==null?void 0:t.speed)??this._getOption("speed"),r=(t==null?void 0:t.defaultTheme)??this._getOption("defaultTheme");let l={...this._animationConfig,autoplay:s?!1:i,loop:typeof e=="number"?!0:e,renderer:this._worker?"svg":this._animationConfig.renderer??"svg"},[d,u,c]=await Promise.all([this._dotLottieLoader.getTheme(r),this._getLottiePlayerInstance(),this._getAudioFactory()]);if(d&&this._animation?this._animation=await Mn(this._animation,d):this._animation=await this._dotLottieLoader.getAnimation(this._currentAnimationId??""),this._activeStateId&&!this._inInteractiveMode){this.enterInteractiveMode(this._activeStateId);return}this.destroy(),this._setPlayerState(()=>({defaultTheme:r,playMode:o,intermission:n,hover:s,loop:e})),c?this._lottie=u.loadAnimation({...l,container:this._container,animationData:this._animation,audioFactory:c}):this._lottie=u.loadAnimation({...l,container:this._container,animationData:this._animation}),typeof this._lottie.resetSegments>"u"&&(this._lottie.resetSegments=()=>{var f;(f=this._lottie)==null||f.playSegments([0,this._lottie.totalFrames],!0)}),this.addEventListeners(),this._container&&(this._container.__lottie=this._lottie),this._setPlayerState(()=>({direction:a,speed:h})),i&&!s&&this.play(),this._updateTestData()}async _getLottiePlayerInstance(){let t=this._animationConfig.renderer??"svg",e;if(this._worker)return t!=="svg"&&x("Worker is only supported with svg renderer. Change or remove renderer prop to get rid of this warning."),e=await W(()=>import("./lottie_worker-WHOD5EO7-64CFFH2E.96a49dc6.js"),["./lottie_worker-WHOD5EO7-64CFFH2E.96a49dc6.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url),e.default;switch(t){case"svg":{this._light?e=await W(()=>import("./lottie_light-IOYK46WR-ECBCRPFC.b3b804e4.js"),["./lottie_light-IOYK46WR-ECBCRPFC.b3b804e4.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url):e=await W(()=>import("./lottie_svg-NPHRPZVQ-H7F42BXO.088570af.js"),["./lottie_svg-NPHRPZVQ-H7F42BXO.088570af.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url);break}case"canvas":{this._light?e=await W(()=>import("./lottie_light_canvas-JOJMVDMY-WR7KP7PE.0f6d71df.js"),["./lottie_light_canvas-JOJMVDMY-WR7KP7PE.0f6d71df.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url):e=await W(()=>import("./lottie_canvas-MBHU2YSW-QKDI7L4D.931a4551.js"),["./lottie_canvas-MBHU2YSW-QKDI7L4D.931a4551.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url);break}case"html":{this._light?e=await W(()=>import("./lottie_light_html-QXO6XUZK-DGMP7OWN.7b282902.js"),["./lottie_light_html-QXO6XUZK-DGMP7OWN.7b282902.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url):e=await W(()=>import("./lottie_html-DEQZ7JWN-7DH66MZZ.0bf71882.js"),["./lottie_html-DEQZ7JWN-7DH66MZZ.0bf71882.js","./chunk-U2GK2GFR.2d4e7f2d.js"],import.meta.url);break}default:throw new Error(`Invalid renderer: ${t}`)}return e.default}_getActiveAnimationId(){var e;let t=this._dotLottieLoader.manifest;return this._activeAnimationId??(t==null?void 0:t.activeAnimationId)??((e=t==null?void 0:t.animations[0])==null?void 0:e.id)??void 0}async load(t){if(this._currentState==="loading"){x("Loading in progress..");return}try{if(this.setCurrentState("loading"),typeof this._src=="string")if(In(this._src)){let i=JSON.parse(this._src);this._dotLottieLoader.loadFromLottieJSON(i)}else{let i=new URL(this._src,window.location.href);await this._dotLottieLoader.loadFromUrl(i.toString())}else if(typeof this._src=="object"&&qt(this._src))this._dotLottieLoader.loadFromLottieJSON(this._src);else throw C("Invalid src provided");if(!this._dotLottieLoader.manifest)throw C("No manifest found");let e=this._getActiveAnimationId();if(!e)throw C("No active animation found");await this._setCurrentAnimation(e),await this.render(t)}catch(e){this.setCurrentState("error"),e instanceof Error&&ft(`Error loading animation: ${e.message}`)}}setErrorState(t){this.setCurrentState("error"),ft(t)}_requireValidDirection(t){if(t!==-1&&t!==1)throw C("Direction can only be -1 (backwards) or 1 (forwards)")}_requireValidIntermission(t){if(t<0||!Number.isInteger(t))throw C("intermission must be a positive number")}_requireValidLoop(t){if(typeof t=="number"&&(!Number.isInteger(t)||t<0))throw C("loop must be a positive number or boolean")}_requireValidSpeed(t){if(typeof t!="number")throw C("speed must be a number")}_requireValidBackground(t){if(typeof t!="string")throw C("background must be a string")}_requireValidAutoplay(t){if(typeof t!="boolean")throw C("autoplay must be a boolean")}_requireValidPlaybackOptions(t){t.direction&&this._requireValidDirection(t.direction),t.intermission&&this._requireValidIntermission(t.intermission),t.loop&&this._requireValidLoop(t.loop),t.speed&&this._requireValidSpeed(t.speed)}},Dn=Object.defineProperty,Vn=Object.getOwnPropertyDescriptor,T=(t,e,i,o)=>{for(var n=o>1?void 0:o?Vn(e,i):e,s=t.length-1,a;s>=0;s--)(a=t[s])&&(n=(o?a(e,i,n):a(n))||n);return o&&n&&Dn(e,i,n),n},Fn=(t,e)=>e.kind==="method"&&e.descriptor&&!("value"in e.descriptor)?{...e,finisher(i){i.createProperty(e.key,t)}}:{kind:"field",key:Symbol(),placement:"own",descriptor:{},originalKey:e.key,initializer(){typeof e.initializer=="function"&&(this[e.key]=e.initializer.call(this))},finisher(i){i.createProperty(e.key,t)}},Bn=(t,e,i)=>{e.constructor.createProperty(i,t)};function N(t){return(e,i)=>i!==void 0?Bn(t,e,i):Fn(t,e)}function Hn(t){return N({...t,state:!0})}var qn=({finisher:t,descriptor:e})=>(i,o)=>{var n;if(o===void 0){let s=(n=i.originalKey)!==null&&n!==void 0?n:i.key,a=e!=null?{kind:"method",placement:"prototype",key:s,descriptor:e(i.key)}:{...i,key:s};return t!=null&&(a.finisher=function(h){t(h,s)}),a}{let s=i.constructor;e!==void 0&&Object.defineProperty(i,o,e(o)),t==null||t(s,o)}};function Rn(t,e){return qn({descriptor:i=>{let o={get(){var n,s;return(s=(n=this.renderRoot)===null||n===void 0?void 0:n.querySelector(t))!==null&&s!==void 0?s:null},enumerable:!0,configurable:!0};if(e){let n=typeof i=="symbol"?Symbol():"__"+i;o.get=function(){var s,a;return this[n]===void 0&&(this[n]=(a=(s=this.renderRoot)===null||s===void 0?void 0:s.querySelector(t))!==null&&a!==void 0?a:null),this[n]}}return o}})}var pe;((pe=window.HTMLSlotElement)===null||pe===void 0?void 0:pe.prototype.assignedElements)!=null;var Zn={name:"@dotlottie/player-component",version:"2.7.0",description:"dotLottie animation player web component.",repository:"https://github.com/dotlottie/player-component.git",homepage:"https://dotlottie.com/players",bugs:"https://github.com/dotlottie/player-component/issues",author:"Jawish Hameed <jawish@lottiefiles.com>",license:"MIT",main:"dist/dotlottie-player.js",module:"dist/dotlottie-player.mjs",types:"dist/dotlottie-player.d.ts",files:["dist"],keywords:["dotlottie","animation","web component","component","lit-element","player"],scripts:{build:"tsup","cypress:open":"cypress open --component",dev:"tsup --watch",lint:"eslint .","lint:fix":"eslint --fix",test:"cypress run --component","type-check":"tsc --noEmit"},dependencies:{lit:"^2.7.5"},devDependencies:{"@dotlottie/common":"workspace:^","@vitejs/plugin-legacy":"^4.1.0","axe-core":"^4.7.2",cypress:"^12.11.0","cypress-axe":"^1.4.0","cypress-ct-lit":"^0.3.2","lottie-web":"^5.12.2",terser:"^5.19.0",tsup:"^6.1.3",typescript:"^4.7.4",vite:"^4.3.9"},publishConfig:{access:"public"},browserslist:["> 3%"]},ai="dotlottie-player",P=class extends At{constructor(){super(...arguments);p(this,"defaultTheme","");p(this,"container");p(this,"playMode",Di.Normal);p(this,"autoplay",!1);p(this,"background","transparent");p(this,"controls",!1);p(this,"direction",1);p(this,"hover",!1);p(this,"loop");p(this,"renderer","svg");p(this,"speed",1);p(this,"src");p(this,"intermission",0);p(this,"activeAnimationId",null);p(this,"light",!1);p(this,"worker",!1);p(this,"activeStateId");p(this,"_seeker",0);p(this,"_dotLottieCommonPlayer");p(this,"_io");p(this,"_loop");p(this,"_renderer","svg");p(this,"_unsubscribeListeners");p(this,"_hasMultipleAnimations",!1);p(this,"_hasMultipleThemes",!1);p(this,"_hasMultipleStates",!1);p(this,"_popoverIsOpen",!1);p(this,"_animationsTabIsOpen",!1);p(this,"_statesTabIsOpen",!1);p(this,"_styleTabIsOpen",!1);p(this,"_themesForCurrentAnimation",[]);p(this,"_statesForCurrentAnimation",[])}_parseLoop(e){let i=parseInt(e,10);return Number.isInteger(i)&&i>0?(this._loop=i,i):typeof e=="string"&&["true","false"].includes(e)?(this._loop=e==="true",this._loop):(x("loop must be a positive integer or a boolean"),!1)}_handleSeekChange(e){let i=e.currentTarget;try{let o=parseInt(i.value,10);if(!this._dotLottieCommonPlayer)return;let n=o/100*this._dotLottieCommonPlayer.totalFrames;this.seek(n)}catch{throw C("Error while seeking animation")}}_initListeners(){let e=this._dotLottieCommonPlayer;if(e===void 0){x("player not initialized - cannot add event listeners","dotlottie-player-component");return}this._unsubscribeListeners=e.state.subscribe((i,o)=>{this._seeker=i.seeker,this.requestUpdate(),o.currentState!==i.currentState&&this.dispatchEvent(new CustomEvent(i.currentState)),this.dispatchEvent(new CustomEvent(ot.Frame,{detail:{frame:i.frame,seeker:i.seeker}})),this.dispatchEvent(new CustomEvent(ot.VisibilityChange,{detail:{visibilityPercentage:i.visibilityPercentage}}))}),e.addEventListener("complete",()=>{this.dispatchEvent(new CustomEvent(ot.Complete))}),e.addEventListener("loopComplete",()=>{this.dispatchEvent(new CustomEvent(ot.LoopComplete))}),e.addEventListener("DOMLoaded",()=>{let i=this.getManifest();i&&i.themes&&(this._themesForCurrentAnimation=i.themes.filter(o=>o.animations.includes(this.getCurrentAnimationId()||""))),i&&i.states&&(this._hasMultipleStates=i.states.length>0,this._statesForCurrentAnimation=[],i.states.forEach(o=>{this._statesForCurrentAnimation.push(o)})),this.dispatchEvent(new CustomEvent(ot.Ready))}),e.addEventListener("data_ready",()=>{this.dispatchEvent(new CustomEvent(ot.DataReady))}),e.addEventListener("data_failed",()=>{this.dispatchEvent(new CustomEvent(ot.DataFail))}),window&&window.addEventListener("click",i=>this._clickOutListener(i))}async load(e,i,o){if(!this.shadowRoot)return;this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.destroy(),this._dotLottieCommonPlayer=new si(e,this.container,{rendererSettings:i??{scaleMode:"noScale",clearCanvas:!0,progressiveLoad:!0,hideOnTransparent:!0},hover:this.hasAttribute("hover")?this.hover:void 0,renderer:this.hasAttribute("renderer")?this._renderer:void 0,loop:this.hasAttribute("loop")?this._loop:void 0,direction:this.hasAttribute("direction")?this.direction===1?1:-1:void 0,speed:this.hasAttribute("speed")?this.speed:void 0,intermission:this.hasAttribute("intermission")?Number(this.intermission):void 0,playMode:this.hasAttribute("playMode")?this.playMode:void 0,autoplay:this.hasAttribute("autoplay")?this.autoplay:void 0,activeAnimationId:this.hasAttribute("activeAnimationId")?this.activeAnimationId:void 0,defaultTheme:this.hasAttribute("defaultTheme")?this.defaultTheme:void 0,light:this.light,worker:this.worker,activeStateId:this.hasAttribute("activeStateId")?this.activeStateId:void 0}),await this._dotLottieCommonPlayer.load(o);let n=this.getManifest();this._hasMultipleAnimations=this.animationCount()>1,n&&(n.themes&&(this._themesForCurrentAnimation=n.themes.filter(s=>s.animations.includes(this.getCurrentAnimationId()||"")),this._hasMultipleThemes=n.themes.length>0),n.states&&(this._hasMultipleStates=n.states.length>0,this._statesForCurrentAnimation=[],n.states.forEach(s=>{this._statesForCurrentAnimation.push(s)}))),this._initListeners()}getCurrentAnimationId(){var e;return(e=this._dotLottieCommonPlayer)==null?void 0:e.currentAnimationId}animationCount(){var e;return this._dotLottieCommonPlayer&&((e=this._dotLottieCommonPlayer.getManifest())==null?void 0:e.animations.length)||0}animations(){var e;return this._dotLottieCommonPlayer?((e=this._dotLottieCommonPlayer.getManifest())==null?void 0:e.animations.map(i=>i.id))||[]:[]}currentAnimation(){return!this._dotLottieCommonPlayer||!this._dotLottieCommonPlayer.currentAnimationId?"":this._dotLottieCommonPlayer.currentAnimationId}getState(){return this._dotLottieCommonPlayer?this._dotLottieCommonPlayer.getState():Vi}getManifest(){var e;return(e=this._dotLottieCommonPlayer)==null?void 0:e.getManifest()}getLottie(){var e;return(e=this._dotLottieCommonPlayer)==null?void 0:e.getAnimationInstance()}getVersions(){return{lottieWebVersion:si.getLottieWebVersion(),dotLottiePlayerVersion:`${Zn.version}`}}previous(e){var i;(i=this._dotLottieCommonPlayer)==null||i.previous(e)}next(e){var i;(i=this._dotLottieCommonPlayer)==null||i.next(e)}reset(){var e;(e=this._dotLottieCommonPlayer)==null||e.reset()}play(e,i){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.play(e,i)}pause(){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.pause()}stop(){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.stop()}playOnShow(e){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.playOnShow(e)}stopPlayOnShow(){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.stopPlayOnShow()}playOnScroll(e){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.playOnScroll(e)}stopPlayOnScroll(){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.stopPlayOnScroll()}seek(e){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.seek(e)}snapshot(e=!0){if(!this.shadowRoot)return"";let i=this.shadowRoot.querySelector(".animation svg"),o=new XMLSerializer().serializeToString(i);if(e){let n=document.createElement("a");n.href=`data:image/svg+xml;charset=utf-8,${encodeURIComponent(o)}`,n.download=`download_${this._seeker}.svg`,document.body.appendChild(n),n.click(),document.body.removeChild(n)}return o}setTheme(e){var i;(i=this._dotLottieCommonPlayer)==null||i.setDefaultTheme(e)}themes(){var e,i;return this._dotLottieCommonPlayer?((i=(e=this._dotLottieCommonPlayer.getManifest())==null?void 0:e.themes)==null?void 0:i.map(o=>o.id))||[]:[]}getDefaultTheme(){return this._dotLottieCommonPlayer?this._dotLottieCommonPlayer.defaultTheme:""}getActiveStateMachine(){return this._dotLottieCommonPlayer?this._dotLottieCommonPlayer.activeStateId:""}_freeze(){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.freeze()}setSpeed(e=1){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.setSpeed(e)}setDirection(e){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.setDirection(e)}setLooping(e){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.setLoop(e)}isLooping(){return this._dotLottieCommonPlayer?this._dotLottieCommonPlayer.loop:!1}togglePlay(){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.togglePlay()}toggleLooping(){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.toggleLoop()}setPlayMode(e){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.setMode(e)}enterInteractiveMode(e){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.enterInteractiveMode(e)}exitInteractiveMode(){this._dotLottieCommonPlayer&&this._dotLottieCommonPlayer.exitInteractiveMode()}revertToManifestValues(e){var i;(i=this._dotLottieCommonPlayer)==null||i.revertToManifestValues(e)}static get styles(){return vo}async firstUpdated(){var e;this.container=(e=this.shadowRoot)==null?void 0:e.querySelector("#animation"),"IntersectionObserver"in window&&(this._io=new IntersectionObserver(i=>{var o,n;i[0]!==void 0&&i[0].isIntersecting?((o=this._dotLottieCommonPlayer)==null?void 0:o.currentState)===vt.Frozen&&this.play():((n=this._dotLottieCommonPlayer)==null?void 0:n.currentState)===vt.Playing&&this._freeze()}),this._io.observe(this.container)),this.loop?this._parseLoop(this.loop):this.hasAttribute("loop")&&this._parseLoop("true"),this.renderer==="svg"?this._renderer="svg":this.renderer==="canvas"?this._renderer="canvas":this.renderer==="html"&&(this._renderer="html"),this.src&&await this.load(this.src)}disconnectedCallback(){var e,i;this._io&&(this._io.disconnect(),this._io=void 0),(e=this._dotLottieCommonPlayer)==null||e.destroy(),(i=this._unsubscribeListeners)==null||i.call(this),window&&window.removeEventListener("click",o=>this._clickOutListener(o))}_clickOutListener(e){!e.composedPath().some(i=>i instanceof HTMLElement?i.classList.contains("popover")||i.id==="lottie-animation-options":!1)&&this._popoverIsOpen&&(this._popoverIsOpen=!1,this.requestUpdate())}renderControls(){var o,n,s,a,h;let e=((o=this._dotLottieCommonPlayer)==null?void 0:o.currentState)===vt.Playing,i=((n=this._dotLottieCommonPlayer)==null?void 0:n.currentState)===vt.Paused;return w`
      <div id="lottie-controls" aria-label="lottie-animation-controls" class="toolbar">
        ${this._hasMultipleAnimations?w`
              <button @click=${()=>this.previous()} aria-label="Previous animation" class="btn-spacing-left">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                    d="M1.69214 13.5C1.69214 13.7761 1.916 14 2.19214 14C2.46828 14 2.69214 13.7761 2.69214 13.5L2.69214 2.5C2.69214 2.22386 2.46828 2 2.19214 2C1.916 2 1.69214 2.22386 1.69214 2.5V13.5ZM12.5192 13.7828C13.1859 14.174 14.0254 13.6933 14.0254 12.9204L14.0254 3.0799C14.0254 2.30692 13.1859 1.8262 12.5192 2.21747L4.13612 7.13769C3.47769 7.52414 3.47769 8.4761 4.13612 8.86255L12.5192 13.7828Z"
                    fill="#20272C"
                  />
                </svg>
              </button>
            `:w``}
        <button
          id="lottie-play-button"
          @click=${()=>{this.togglePlay()}}
          class=${e||i?`active ${this._hasMultipleAnimations?"btn-spacing-center":"btn-spacing-right"}`:`${this._hasMultipleAnimations?"btn-spacing-center":"btn-spacing-right"}`}
          aria-label="play / pause animation"
        >
          ${e?w`
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M3.99996 2C3.26358 2 2.66663 2.59695 2.66663 3.33333V12.6667C2.66663 13.403 3.26358 14 3.99996 14H5.33329C6.06967 14 6.66663 13.403 6.66663 12.6667V3.33333C6.66663 2.59695 6.06967 2 5.33329 2H3.99996Z"
                    fill="#20272C"
                  />
                  <path
                    d="M10.6666 2C9.93025 2 9.33329 2.59695 9.33329 3.33333V12.6667C9.33329 13.403 9.93025 14 10.6666 14H12C12.7363 14 13.3333 13.403 13.3333 12.6667V3.33333C13.3333 2.59695 12.7363 2 12 2H10.6666Z"
                    fill="#20272C"
                  />
                </svg>
              `:w`
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M3.33337 3.46787C3.33337 2.52312 4.35948 1.93558 5.17426 2.41379L12.8961 6.94592C13.7009 7.41824 13.7009 8.58176 12.8961 9.05408L5.17426 13.5862C4.35948 14.0644 3.33337 13.4769 3.33337 12.5321V3.46787Z"
                    fill="#20272C"
                  />
                </svg>
              `}
        </button>
        ${this._hasMultipleAnimations?w`
              <button @click=${()=>this.next()} aria-label="Next animation" class="btn-spacing-right">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                    d="M14.3336 2.5C14.3336 2.22386 14.1097 2 13.8336 2C13.5574 2 13.3336 2.22386 13.3336 2.5V13.5C13.3336 13.7761 13.5574 14 13.8336 14C14.1097 14 14.3336 13.7761 14.3336 13.5V2.5ZM3.50618 2.21722C2.83954 1.82595 2 2.30667 2 3.07965V12.9201C2 13.6931 2.83954 14.1738 3.50618 13.7825L11.8893 8.86231C12.5477 8.47586 12.5477 7.52389 11.8893 7.13745L3.50618 2.21722Z"
                    fill="#20272C"
                  />
                </svg>
              </button>
            `:w``}
        <input
          id="lottie-seeker-input"
          class="seeker ${((s=this._dotLottieCommonPlayer)==null?void 0:s.direction)===-1?"to-left":""}"
          type="range"
          min="0"
          step="1"
          max="100"
          .value=${this._seeker}
          @input=${r=>this._handleSeekChange(r)}
          @mousedown=${()=>{this._freeze()}}
          @mouseup=${()=>{var r;(r=this._dotLottieCommonPlayer)==null||r.unfreeze()}}
          aria-valuemin="1"
          aria-valuemax="100"
          role="slider"
          aria-valuenow=${this._seeker}
          aria-label="lottie-seek-input"
          style=${`--seeker: ${this._seeker}`}
        />
        <button
          id="lottie-loop-toggle"
          @click=${()=>this.toggleLooping()}
          class=${(a=this._dotLottieCommonPlayer)!=null&&a.loop?"active btn-spacing-left":"btn-spacing-left"}
          aria-label="loop-toggle"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M10.8654 2.31319C11.0607 2.11793 11.3772 2.11793 11.5725 2.31319L13.4581 4.19881C13.6534 4.39407 13.6534 4.71066 13.4581 4.90592L11.5725 6.79154C11.3772 6.9868 11.0607 6.9868 10.8654 6.79154C10.6701 6.59628 10.6701 6.27969 10.8654 6.08443L11.6162 5.33362H4V6.66695C4 7.03514 3.70152 7.33362 3.33333 7.33362C2.96514 7.33362 2.66666 7.03514 2.66666 6.66695L2.66666 4.66695C2.66666 4.29876 2.96514 4.00028 3.33333 4.00028H11.8454L10.8654 3.0203C10.6701 2.82504 10.6701 2.50846 10.8654 2.31319Z"
              fill="currentColor"
            />
            <path
              d="M12.4375 11.9999C12.8057 11.9999 13.1042 11.7014 13.1042 11.3332V9.33321C13.1042 8.96502 12.8057 8.66655 12.4375 8.66655C12.0693 8.66655 11.7708 8.96502 11.7708 9.33321V10.6665H4.15462L4.90543 9.91573C5.10069 9.72047 5.10069 9.40389 4.90543 9.20862C4.71017 9.01336 4.39359 9.01336 4.19832 9.20862L2.31271 11.0942C2.11744 11.2895 2.11744 11.6061 2.31271 11.8013L4.19832 13.687C4.39359 13.8822 4.71017 13.8822 4.90543 13.687C5.10069 13.4917 5.10069 13.1751 4.90543 12.9799L3.92545 11.9999H12.4375Z"
              fill="currentColor"
            />
          </svg>
        </button>
        ${this._hasMultipleAnimations||this._hasMultipleThemes||this._hasMultipleStates?w`
              <button
                id="lottie-animation-options"
                @click=${()=>{this._popoverIsOpen=!this._popoverIsOpen,this.requestUpdate()}}
                aria-label="options"
                class="btn-spacing-right"
                style=${`background-color: ${this._popoverIsOpen?"var(--lottie-player-toolbar-icon-hover-color)":""}`}
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M8.33337 11.6666C7.78109 11.6666 7.33337 12.1143 7.33337 12.6666C7.33337 13.2189 7.78109 13.6666 8.33337 13.6666C8.88566 13.6666 9.33337 13.2189 9.33337 12.6666C9.33337 12.1143 8.88566 11.6666 8.33337 11.6666Z"
                    fill="#20272C"
                  />
                  <path
                    d="M7.33337 7.99992C7.33337 7.44763 7.78109 6.99992 8.33337 6.99992C8.88566 6.99992 9.33338 7.44763 9.33338 7.99992C9.33338 8.5522 8.88566 8.99992 8.33337 8.99992C7.78109 8.99992 7.33337 8.5522 7.33337 7.99992Z"
                    fill="#20272C"
                  />
                  <path
                    d="M7.33337 3.33325C7.33337 2.78097 7.78109 2.33325 8.33337 2.33325C8.88566 2.33325 9.33338 2.78097 9.33338 3.33325C9.33338 3.88554 8.88566 4.33325 8.33337 4.33325C7.78109 4.33325 7.33337 3.88554 7.33337 3.33325Z"
                    fill="#20272C"
                  />
                </svg>
              </button>
            `:w``}
      </div>
      ${this._popoverIsOpen?w`
            <div
              id="popover"
              class="popover"
              tabindex="0"
              aria-label="lottie animations themes popover"
              style="min-height: ${this.themes().length>0?"84px":"auto"}"
            >
              ${!this._animationsTabIsOpen&&!this._styleTabIsOpen&&!this._statesTabIsOpen?w`
                    <button
                      class="popover-button"
                      tabindex="0"
                      aria-label="animations"
                      @click=${()=>{this._animationsTabIsOpen=!this._animationsTabIsOpen,this.requestUpdate()}}
                      @keydown=${r=>{(r.code==="Space"||r.code==="Enter")&&(this._animationsTabIsOpen=!this._animationsTabIsOpen,this.requestUpdate())}}
                    >
                      <div class="popover-button-text">Animations</div>
                      <div>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path
                            fill-rule="evenodd"
                            clip-rule="evenodd"
                            d="M10.4697 17.5303C10.1768 17.2374 10.1768 16.7626 10.4697 16.4697L14.9393 12L10.4697 7.53033C10.1768 7.23744 10.1768 6.76256 10.4697 6.46967C10.7626 6.17678 11.2374 6.17678 11.5303 6.46967L16.5303 11.4697C16.8232 11.7626 16.8232 12.2374 16.5303 12.5303L11.5303 17.5303C11.2374 17.8232 10.7626 17.8232 10.4697 17.5303Z"
                            fill="#4C5863"
                          />
                        </svg>
                      </div>
                    </button>
                  `:w``}
              ${this._hasMultipleThemes&&!this._styleTabIsOpen&&!this._animationsTabIsOpen&&!this._statesTabIsOpen?w` <button
                    class="popover-button"
                    aria-label="Themes"
                    @click=${()=>{this._styleTabIsOpen=!this._styleTabIsOpen,this.requestUpdate()}}
                    @keydown=${r=>{(r.code==="Space"||r.code==="Enter")&&(this._styleTabIsOpen=!this._styleTabIsOpen,this.requestUpdate())}}
                  >
                    <div class="popover-button-text">Themes</div>
                    <div>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                          d="M10.4697 17.5303C10.1768 17.2374 10.1768 16.7626 10.4697 16.4697L14.9393 12L10.4697 7.53033C10.1768 7.23744 10.1768 6.76256 10.4697 6.46967C10.7626 6.17678 11.2374 6.17678 11.5303 6.46967L16.5303 11.4697C16.8232 11.7626 16.8232 12.2374 16.5303 12.5303L11.5303 17.5303C11.2374 17.8232 10.7626 17.8232 10.4697 17.5303Z"
                          fill="#4C5863"
                        />
                      </svg>
                    </div>
                  </button>`:""}
              ${this._hasMultipleStates&&!this._styleTabIsOpen&&!this._animationsTabIsOpen&&!this._statesTabIsOpen?w` <button
                    class="popover-button"
                    aria-label="States"
                    @click=${()=>{this._statesTabIsOpen=!this._statesTabIsOpen,this.requestUpdate()}}
                    @keydown=${r=>{(r.code==="Space"||r.code==="Enter")&&(this._statesTabIsOpen=!this._statesTabIsOpen,this.requestUpdate())}}
                  >
                    <div class="popover-button-text">States</div>
                    <div>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                          d="M10.4697 17.5303C10.1768 17.2374 10.1768 16.7626 10.4697 16.4697L14.9393 12L10.4697 7.53033C10.1768 7.23744 10.1768 6.76256 10.4697 6.46967C10.7626 6.17678 11.2374 6.17678 11.5303 6.46967L16.5303 11.4697C16.8232 11.7626 16.8232 12.2374 16.5303 12.5303L11.5303 17.5303C11.2374 17.8232 10.7626 17.8232 10.4697 17.5303Z"
                          fill="#4C5863"
                        />
                      </svg>
                    </div>
                  </button>`:""}
              ${this._animationsTabIsOpen?w`<button
                      class="option-title-button"
                      aria-label="Back to main popover menu"
                      @click=${()=>{this._animationsTabIsOpen=!this._animationsTabIsOpen,this.requestUpdate()}}
                    >
                      <div class="option-title-chevron">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path
                            fill-rule="evenodd"
                            clip-rule="evenodd"
                            d="M13.5303 6.46967C13.8232 6.76256 13.8232 7.23744 13.5303 7.53033L9.06066 12L13.5303 16.4697C13.8232 16.7626 13.8232 17.2374 13.5303 17.5303C13.2374 17.8232 12.7626 17.8232 12.4697 17.5303L7.46967 12.5303C7.17678 12.2374 7.17678 11.7626 7.46967 11.4697L12.4697 6.46967C12.7626 6.17678 13.2374 6.17678 13.5303 6.46967Z"
                            fill="#20272C"
                          />
                        </svg>
                      </div>
                      <div>Animations</div>
                    </button>
                    <div class="option-title-separator"></div>
                    <div class="option-row">
                      <ul>
                        ${this.animations().map(r=>w`
                            <li>
                              <button
                                class="option-button"
                                aria-label=${`${r}`}
                                @click=${()=>{this._animationsTabIsOpen=!this._animationsTabIsOpen,this._popoverIsOpen=!this._popoverIsOpen,this.play(r),this.requestUpdate()}}
                                @keydown=${l=>{(l.code==="Space"||l.code==="Enter")&&(this._animationsTabIsOpen=!this._animationsTabIsOpen,this._popoverIsOpen=!this._popoverIsOpen,this.play(r),this.requestUpdate())}}
                              >
                                <div class="option-tick">
                                  ${this.currentAnimation()===r?w`
                                        <svg
                                          width="24"
                                          height="24"
                                          viewBox="0 0 24 24"
                                          fill="none"
                                          xmlns="http://www.w3.org/2000/svg"
                                        >
                                          <path
                                            fill-rule="evenodd"
                                            clip-rule="evenodd"
                                            d="M20.5281 5.9372C20.821 6.23009 20.821 6.70497 20.5281 6.99786L9.46297 18.063C9.32168 18.2043 9.12985 18.2833 8.93004 18.2826C8.73023 18.2819 8.53895 18.2015 8.39864 18.0593L3.46795 13.0596C3.1771 12.7647 3.1804 12.2898 3.47532 11.999C3.77024 11.7081 4.2451 11.7114 4.53595 12.0063L8.93634 16.4683L19.4675 5.9372C19.7604 5.64431 20.2352 5.64431 20.5281 5.9372Z"
                                            fill="#20272C"
                                          />
                                        </svg>
                                      `:w`<div style="width: 24px; height: 24px"></div>`}
                                </div>
                                <div>${r}</div>
                              </button>
                            </li>
                          `)}
                      </ul>
                    </div> `:w``}
              ${this._styleTabIsOpen?w`<div class="option-title-themes-row">
                      <button
                        class="option-title-button themes"
                        aria-label="Back to main popover menu"
                        @click=${()=>{this._styleTabIsOpen=!this._styleTabIsOpen,this.requestUpdate()}}
                      >
                        <div class="option-title-chevron">
                          <svg
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <path
                              fill-rule="evenodd"
                              clip-rule="evenodd"
                              d="M13.5303 6.46967C13.8232 6.76256 13.8232 7.23744 13.5303 7.53033L9.06066 12L13.5303 16.4697C13.8232 16.7626 13.8232 17.2374 13.5303 17.5303C13.2374 17.8232 12.7626 17.8232 12.4697 17.5303L7.46967 12.5303C7.17678 12.2374 7.17678 11.7626 7.46967 11.4697L12.4697 6.46967C12.7626 6.17678 13.2374 6.17678 13.5303 6.46967Z"
                              fill="#20272C"
                            />
                          </svg>
                        </div>
                        <div class="option-title-text">Themes</div>
                        ${((h=this._dotLottieCommonPlayer)==null?void 0:h.defaultTheme)===""?w``:w`
                              <button
                                class="reset-btn"
                                @click=${()=>{this.setTheme(""),this.requestUpdate()}}
                              >
                                Reset
                              </button>
                            `}
                      </button>
                    </div>
                    <div class="option-title-separator"></div>
                    <div class="option-row">
                      <ul>
                        ${this._themesForCurrentAnimation.map(r=>w`
                            <li>
                              <button
                                class="option-button"
                                aria-label="${r.id}"
                                @click=${()=>{this.setTheme(r.id)}}
                                @keydown=${l=>{(l.code==="Space"||l.code==="Enter")&&this.setTheme(r.id)}}
                              >
                                <div class="option-tick">
                                  ${this.getDefaultTheme()===r.id?w`
                                        <svg
                                          width="24"
                                          height="24"
                                          viewBox="0 0 24 24"
                                          fill="none"
                                          xmlns="http://www.w3.org/2000/svg"
                                        >
                                          <path
                                            fill-rule="evenodd"
                                            clip-rule="evenodd"
                                            d="M20.5281 5.9372C20.821 6.23009 20.821 6.70497 20.5281 6.99786L9.46297 18.063C9.32168 18.2043 9.12985 18.2833 8.93004 18.2826C8.73023 18.2819 8.53895 18.2015 8.39864 18.0593L3.46795 13.0596C3.1771 12.7647 3.1804 12.2898 3.47532 11.999C3.77024 11.7081 4.2451 11.7114 4.53595 12.0063L8.93634 16.4683L19.4675 5.9372C19.7604 5.64431 20.2352 5.64431 20.5281 5.9372Z"
                                            fill="#20272C"
                                          />
                                        </svg>
                                      `:w`<div style="width: 24px; height: 24px"></div>`}
                                </div>
                                <div>${r.id}</div>
                              </button>
                            </li>
                          `)}
                      </ul>
                    </div>`:w``}
              ${this._statesTabIsOpen?w`<div class="option-title-themes-row">
                      <button
                        class="option-title-button themes"
                        aria-label="Back to main popover menu"
                        @click=${()=>{this._statesTabIsOpen=!this._statesTabIsOpen,this.requestUpdate()}}
                      >
                        <div class="option-title-chevron">
                          <svg
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <path
                              fill-rule="evenodd"
                              clip-rule="evenodd"
                              d="M13.5303 6.46967C13.8232 6.76256 13.8232 7.23744 13.5303 7.53033L9.06066 12L13.5303 16.4697C13.8232 16.7626 13.8232 17.2374 13.5303 17.5303C13.2374 17.8232 12.7626 17.8232 12.4697 17.5303L7.46967 12.5303C7.17678 12.2374 7.17678 11.7626 7.46967 11.4697L12.4697 6.46967C12.7626 6.17678 13.2374 6.17678 13.5303 6.46967Z"
                              fill="#20272C"
                            />
                          </svg>
                        </div>
                        <div class="option-title-text">States</div>
                        <button
                          class="reset-btn"
                          @click=${()=>{this.exitInteractiveMode(),this.requestUpdate()}}
                        >
                          Reset
                        </button>
                      </button>
                    </div>
                    <div class="option-title-separator"></div>
                    <div class="option-row">
                      <ul>
                        ${this._statesForCurrentAnimation.map(r=>w`
                            <li>
                              <button
                                class="option-button"
                                aria-label="${r}"
                                @click=${()=>{this.enterInteractiveMode(r)}}
                                @keydown=${l=>{(l.code==="Space"||l.code==="Enter")&&this.enterInteractiveMode(r)}}
                              >
                                <div class="option-tick">
                                  ${this.getActiveStateMachine()===r?w`
                                        <svg
                                          width="24"
                                          height="24"
                                          viewBox="0 0 24 24"
                                          fill="none"
                                          xmlns="http://www.w3.org/2000/svg"
                                        >
                                          <path
                                            fill-rule="evenodd"
                                            clip-rule="evenodd"
                                            d="M20.5281 5.9372C20.821 6.23009 20.821 6.70497 20.5281 6.99786L9.46297 18.063C9.32168 18.2043 9.12985 18.2833 8.93004 18.2826C8.73023 18.2819 8.53895 18.2015 8.39864 18.0593L3.46795 13.0596C3.1771 12.7647 3.1804 12.2898 3.47532 11.999C3.77024 11.7081 4.2451 11.7114 4.53595 12.0063L8.93634 16.4683L19.4675 5.9372C19.7604 5.64431 20.2352 5.64431 20.5281 5.9372Z"
                                            fill="#20272C"
                                          />
                                        </svg>
                                      `:w`<div style="width: 24px; height: 24px"></div>`}
                                </div>
                                <div>${r}</div>
                              </button>
                            </li>
                          `)}
                      </ul>
                    </div>`:w``}
            </div>
          `:w``}
    `}render(){var o;let e=this.controls?"main controls":"main",i=this.controls?"animation controls":"animation";return w`
      <div id="animation-container" class=${e} lang="en" role="img" aria-label="lottie-animation-container">
        <div id="animation" class=${i} style="background:${this.background};">
          ${((o=this._dotLottieCommonPlayer)==null?void 0:o.currentState)===vt.Error?w` <div class="error">⚠️</div> `:void 0}
        </div>
        ${this.controls?this.renderControls():void 0}
      </div>
    `}};T([N({type:String})],P.prototype,"defaultTheme",2),T([Rn("#animation")],P.prototype,"container",2),T([N()],P.prototype,"playMode",2),T([N({type:Boolean})],P.prototype,"autoplay",2),T([N({type:String})],P.prototype,"background",2),T([N({type:Boolean})],P.prototype,"controls",2),T([N({type:Number})],P.prototype,"direction",2),T([N({type:Boolean})],P.prototype,"hover",2),T([N({type:String})],P.prototype,"loop",2),T([N({type:String})],P.prototype,"renderer",2),T([N({type:Number})],P.prototype,"speed",2),T([N({type:String})],P.prototype,"src",2),T([N()],P.prototype,"intermission",2),T([N({type:String})],P.prototype,"activeAnimationId",2),T([N({type:Boolean})],P.prototype,"light",2),T([N({type:Boolean})],P.prototype,"worker",2),T([N({type:String})],P.prototype,"activeStateId",2),T([Hn()],P.prototype,"_seeker",2);customElements.get(ai)||customElements.define(ai,P);/*! Bundled license information:

@lit/reactive-element/decorators/custom-element.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/property.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/state.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/base.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/event-options.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query-all.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query-async.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query-assigned-elements.js:
  (**
   * @license
   * Copyright 2021 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query-assigned-nodes.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)
*/function Fi(t){return!t||t===document.body?!1:getComputedStyle(t).position==="fixed"?!0:Fi(t.parentElement)}function Jn(t){const e=window.scrollY,i=Math.floor(e/t),o=e%t;return i%2===0?o/t*100:100-o/t*100}function Wn(t){const e=window.innerHeight,o=document.documentElement.scrollTop+e,s=t.getBoundingClientRect().top+window.scrollY,a=s+t.offsetHeight+e,h=(o-s)/(a-s)*100;return Math.max(Math.min(h,100),0)}function fe(t,e){return Math.round(t*(e/100))}function Kn(t,e){const i=me(null),o=new Map;function n(){if(h(),i.value=null,!e.value){i.value=t.value;return}const r=document.querySelectorAll(e.value);r.length>0&&r[0]instanceof HTMLElement&&(i.value=r[0]);for(const l of r)if(l instanceof HTMLElement&&l.contains(t.value)){i.value=l;break}i.value||(i.value=t.value)}function s(r,l,d=!1){var u;(u=i.value)==null||u.addEventListener(r,l,d),o.set(r,[l,d])}function a(r,l,d=!1){var u;(u=i.value)==null||u.removeEventListener(r,l,d),o.delete(r)}function h(){for(const[r,[l,d]]of o)a(r,l,d)}return li(e,()=>{n()}),hi(()=>{n()}),$e(()=>{h()}),{lottieContainerRef:i,initContainer:n,addEventListener:s,removeAllEventListener:h}}const Yn=1e3;function Xn(t,e,i,o,n,s){const{lottieContainerRef:a,initContainer:h,addEventListener:r}=Kn(t,i),l=()=>{if(!e.value)return;const y=e.value.getLottie().totalFrames;function b(S){if(!(S instanceof MouseEvent)||!a.value)return;const $=a.value.getBoundingClientRect();let I=0;if(s.value==="vertical"?I=(S.clientY-$.top)/$.height*100:I=(S.clientX-$.left)/$.width*100,!e.value)return;const k=fe(y,I);e.value.seek(k)}r("mousemove",b)},d=()=>{function y(){e.value&&(e.value.setDirection(1),e.value.play())}function b(){e.value&&(e.value.setDirection(-1),e.value.play())}r("mouseenter",y),r("mouseleave",b)},u=()=>{function y(){var S;(S=e.value)==null||S.play()}function b(){var S;(S=e.value)==null||S.stop()}r("mouseenter",y),r("mouseleave",b)},c=()=>{if(!e.value)return;const y=e.value.getLottie().totalFrames;function b(){if(!e.value)return;const $=Jn(Yn),I=fe(y,$);e.value.seek(I)}document.addEventListener("scroll",b);function S(){document.removeEventListener("scroll",b)}return S},f=()=>{if(!e.value||!a.value)return;let y=!1;const b=e.value.getLottie().totalFrames,S=new IntersectionObserver(([O])=>{O.isIntersecting&&!y?I():!O.isIntersecting&&y&&k(),y=O.isIntersecting},{threshold:0});S.observe(a.value);function $(){if(!e.value||!y||!a.value)return;const O=Wn(a.value),tt=fe(b,O);e.value.seek(tt)}function I(){document.addEventListener("scroll",$)}function k(){document.removeEventListener("scroll",$)}function F(){S&&S.disconnect(),k()}return F},_=()=>{function y(){var b;e.value&&(e.value.seek("0%"),(b=e.value)==null||b.play())}r("click",y)},m=()=>{if(!e.value||!a.value)return;const y=new IntersectionObserver(([S])=>{S.isIntersecting&&setTimeout(()=>{var $;if(e.value){if(!n.value){const I=()=>{b(),e.value&&e.value.removeEventListener("complete",I)};e.value.addEventListener("complete",I)}($=e.value)==null||$.play()}},o.value??0)},{threshold:.3});y.observe(a.value);function b(){y&&y.disconnect()}return b};let v;function g(y){y.playType&&(v&&(v(),v=void 0),h(),y.playType==="scroll"?Fi(t.value)?v=c():v=f():y.playType==="hover"?y.hold?v=d():v=u():y.playType==="cursor"?v=l():y.playType==="click"?v=_():y.playType==="appear"&&(v=m()))}return $e(()=>{v&&v()}),{setEvent:g}}const Gn=Ji({name:"LottieRenderer",props:{...Wi},setup(t){const e=Ki(t),i=e.getDomRef(!0),o=me(null),n=Yi(()=>{var r;return Xi(((r=t.studioDom.content)==null?void 0:r.lottie)??{},t.rm)}),{setEvent:s}=Xn(i,o,Ut(()=>{var r;return(r=n.value)==null?void 0:r.containerId}),Ut(()=>{var r;return(r=n.value)==null?void 0:r.delay}),Ut(()=>{var r;return(r=n.value)==null?void 0:r.loop}),Ut(()=>{var r;return(r=n.value)==null?void 0:r.cursorDirection}));async function a(){if(!i.value)return;i.value.innerHTML="";const r=n.value;if(!r||!r.src||!r.playType)return;const l=document.createElement("dotlottie-player");{l.setAttribute("src",r.src),r.playType==="autoplay"&&l.setAttribute("autoplay","true"),r.controls&&l.setAttribute("controls","true"),r.speed&&l.setAttribute("speed",String(r.speed));const u=r.playType==="hover"&&r.hold,c=!!(r.loop&&!u);l.setAttribute("loop",String(c))}i.value.appendChild(l),o.value=l;const d=()=>{s(r),l.removeEventListener("ready",d)};l.addEventListener("ready",d)}const h=me("a");return li([()=>{var r;return(r=n.value)==null?void 0:r.playType},()=>{var r;return(r=n.value)==null?void 0:r.controls},()=>{var r;return(r=n.value)==null?void 0:r.loop},()=>{var r;return(r=n.value)==null?void 0:r.hold},()=>{var r;return(r=n.value)==null?void 0:r.speed},()=>{var r;return(r=n.value)==null?void 0:r.src},()=>{var r;return(r=n.value)==null?void 0:r.cursorDirection},()=>{var r;return(r=n.value)==null?void 0:r.delay},()=>{var r;return(r=n.value)==null?void 0:r.containerId},()=>{var r;return(r=t.studioDom.style)==null?void 0:r.position},()=>{var r,l;return(l=(r=t.studioDom.style)==null?void 0:r["@small"])==null?void 0:l.position},()=>{var r,l;return(l=(r=t.studioDom.style)==null?void 0:r["@tablet"])==null?void 0:l.position},()=>{var r,l;return(l=(r=t.studioDom.style)==null?void 0:r["@mobile"])==null?void 0:l.position},()=>{var r,l;return(l=(r=t.studioDom.style)==null?void 0:r["@mini"])==null?void 0:l.position}],async()=>{h.value+="a",await Gi(),a()}),hi(()=>{a()}),$e(()=>{var r;(r=o.value)==null||r.disconnectedCallback()}),()=>Qi("div",{key:h.value,ref:i,...e.attrs.value,class:["sd"],style:e.style.value})}}),lr=Object.freeze(Object.defineProperty({__proto__:null,default:Gn},Symbol.toStringTag,{value:"Module"}));export{q as F,lr as L,ar as n,ft as t,C as v};
