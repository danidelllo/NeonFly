import {actionWiring} from "./wiring.js";
import {loadFlyBody} from "./fly-body.js";
import {inputMaps} from "./input-maps.js";
import {rangeRadar} from "./radar.js";
const $=id=>document.getElementById(id), atlas=$("atlas");
const fmt=n=>n.toLocaleString("en-US");
async function get(path){const r=await fetch(path,{cache:"no-store"});if(!r.ok)throw Error(await r.text());return r}
async function run(){
  const gl=atlas.getContext("webgl2",{alpha:true,antialias:false,powerPreference:"high-performance"});
  if(!gl)throw Error("WebGL 2 unavailable; enable browser graphics acceleration.");
  const binary=async(path,Type)=>new Type(await (await get(path)).arrayBuffer());
  const [meta,xyz,bundles,centers,inputs,outputs,rangeNeurons]=await Promise.all([get("/meta").then(r=>r.json()),
    binary("/xyz",Float32Array),binary("/bundles",Int32Array),binary("/centroids",Float32Array),
    binary("/inputs",Uint32Array),binary("/outputs",Uint32Array),binary('/range-neurons',Uint32Array)]);
  if(xyz.length!==3*meta.nodes)throw Error("3D coordinates do not match neuron roster");
  const roles=new Uint8Array(meta.nodes);
  for(const i of inputs)roles[i]|=1;
  for(const i of outputs)roles[i]|=2;
  for(const i of rangeNeurons)roles[i]|=4;
  $('rangeCount').textContent=fmt(rangeNeurons.length);
  const radar=rangeRadar();
  $("inputCount").textContent=fmt(inputs.length);$("outputCount").textContent=fmt(outputs.length);
  for(const key of ["nodes","edges","synapses"])$(key).textContent=fmt(meta[key]);
  function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
  const pr=gl.createProgram();
  gl.attachShader(pr,shader(gl.VERTEX_SHADER,`#version 300 es
  layout(location=0) in vec3 p;layout(location=1) in float v;uniform vec2 scale,pan;uniform mat3 rotation;uniform float point,depth;out float val;
  void main(){vec3 q=rotation*p;gl_Position=vec4((q.xy+pan)*scale,q.z*depth,1.);gl_PointSize=point;val=v;}`));
  gl.attachShader(pr,shader(gl.FRAGMENT_SHADER,`#version 300 es
  precision highp float;in float val;uniform int mode;out vec4 color;
  void main(){if(mode==0){float a=clamp(sqrt(abs(val))*2.7,0.,1.);color=vec4(mix(vec3(.26,.38,.48), val<0.?vec3(.66,.49,1.):vec3(.28,1.,.73),a),.20+.75*a);}
  else if(mode==1)color=vec4(.32,.65,.78,clamp(val,.008,.14));
  else if(mode==5||mode==6){float d=length(gl_PointCoord-.5);if(d<.28||d>.5)discard;color=mode==5?vec4(.29,.76,1.,.76):vec4(1.,.55,.19,.94);}
  else if(mode==7){float d=length(gl_PointCoord-.5);if(d<.28||d>.5)discard;color=vec4(.72,.68,1.,.8);}
  else if(mode==2)color=vec4(.70,.53,1.,.54);else if(mode==3)color=vec4(.29,1.,.73,.54);else color=vec4(1.,.85,.38,1.);}`));
  gl.linkProgram(pr);if(!gl.getProgramParameter(pr,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(pr));gl.useProgram(pr);
  const u=Object.fromEntries(["scale","pan","point","mode","rotation","depth"].map(k=>[k,gl.getUniformLocation(pr,k)]));
  function buf(arr,use=gl.STATIC_DRAW){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,arr,use);return b}
  const pos=buf(xyz), activity=buf(meta.nodes*4,gl.DYNAMIC_DRAW);
  function indexBuffer(indices){const b=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,b);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.STATIC_DRAW);return b}
  const highlights=[[indexBuffer(inputs),inputs.length,"inputs",5],[indexBuffer(outputs),outputs.length,"outputs",6],
                    [indexBuffer(rangeNeurons),rangeNeurons.length,'rangeInputs',7]];
  const lines=[], strength=[];
  for(let i=0;i<bundles.length;i+=3){const a=bundles[i],b=bundles[i+1];if(a===b)continue;
    for(const k of [a,b]){lines.push(...centers.subarray(k*3,k*3+3));strength.push(Math.log1p(bundles[i+2])*.008)}}
  const overview=buf(new Float32Array(lines)), density=buf(new Float32Array(strength));
  const incoming=buf(0,gl.DYNAMIC_DRAW), outgoing=buf(0,gl.DYNAMIC_DRAW);
  let zoom=1,pan=[0,0],sx=1,sy=1,act=null,sel=null,hist=[],frame=null,seq=-1,selectionEpoch=0;
  const homeYaw=.30,homePitch=-.20;
  let yaw=homeYaw,pitch=homePitch,rot=new Float32Array(9);
  let flyBody=null,bodyEnabled=false;
  const emptyInspector=$("identity").textContent;
  const circuit=actionWiring(outputs,(i,w,h)=>{
    const [x,y,z]=xyz.subarray(3*i,3*i+3);
    return [(1+sx*(rot[0]*x+rot[3]*y+rot[6]*z+pan[0]))*w/2,
            (1+sy*(rot[1]*x+rot[4]*y+rot[7]*z+pan[1]))*h/2]
  },paint);
  let sensory=null;
  get('/encoding').then(r=>r.json()).then(mapping=>{
    sensory=inputMaps(mapping,(i,box)=>{
      const [x,y,z]=xyz.subarray(3*i,3*i+3),ab=atlas.getBoundingClientRect();
      return [ab.left-box.left+(1+sx*(rot[0]*x+rot[3]*y+rot[6]*z+pan[0]))*ab.width/2,
              ab.top-box.top+(1+sy*(rot[1]*x+rot[4]*y+rot[7]*z+pan[1]))*ab.height/2]
    },paint);
    if(frame)sensory.sample(frame,act);paint()
  }).catch(e=>{$('mapStatus').textContent=e.message});
  function bind(i,b,n){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.enableVertexAttribArray(i);gl.vertexAttribPointer(i,n,gl.FLOAT,false,0,0)}
  function paint(){
    const box=atlas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,2),w=Math.round(box.width*ratio),h=Math.round(box.height*ratio);
    if(!w||!h)return;if(atlas.width!==w||atlas.height!==h){atlas.width=w;atlas.height=h}
    gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    sx=zoom*Math.min(1,h/w*1.55);sy=sx*w/h;
    const cy=Math.cos(yaw),syaw=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    rot.set([cy,sp*syaw,-cp*syaw,0,cp,sp,syaw,-sp*cy,cp*cy]);
    if(bodyEnabled&&flyBody)flyBody.draw(rot,sx,sy,pan);
    gl.useProgram(pr);gl.uniformMatrix3fv(u.rotation,false,rot);gl.uniform1f(u.depth,flyBody?.depth||.5);
    gl.uniform2f(u.scale,sx,-sy);gl.uniform2fv(u.pan,pan);gl.uniform1f(u.point,Math.max(1,ratio*Math.min(3.3,Math.sqrt(zoom))));
    if($("bundled").checked){bind(0,overview,3);bind(1,density,1);gl.uniform1i(u.mode,1);gl.drawArrays(gl.LINES,0,strength.length)}
    bind(0,pos,3);bind(1,activity,1);gl.enable(gl.DEPTH_TEST);gl.uniform1i(u.mode,0);gl.drawArrays(gl.POINTS,0,meta.nodes);gl.disable(gl.DEPTH_TEST);
    for(const [b,n,id,mode] of highlights)if($(id).checked){
      gl.uniform1i(u.mode,mode);gl.uniform1f(u.point,ratio*Math.min(10,(mode===5?3:4)*Math.sqrt(zoom)));
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,b);gl.drawElements(gl.POINTS,n,gl.UNSIGNED_INT,0)
    }
    if(sel){
      gl.disableVertexAttribArray(1);gl.vertexAttrib1f(1,0);
      for(const [b,name,m] of [[incoming,"in",2],[outgoing,"out",3]]){bind(0,b,3);gl.uniform1i(u.mode,m);gl.drawArrays(gl.LINES,0,sel[name].length*2)}
      bind(0,pos,3);gl.uniform1i(u.mode,4);gl.uniform1f(u.point,8*ratio);gl.drawArrays(gl.POINTS,sel.i,1)
    }
    atlas.dataset.pan=pan.join(',');atlas.dataset.rotation=[yaw,pitch].join(',');
    circuit.draw();sensory?.draw()
  }
  async function select(q){const ticket=++selectionEpoch;try{
    const row=await (await get("/node?"+q)).json();if(ticket!==selectionEpoch)return;sel=row;hist=[];
    for(const [name,b] of [["in",incoming],["out",outgoing]]){
      const arr=new Float32Array(row[name].length*6);row[name].forEach(([j],k)=>{arr.set(xyz.subarray(row.i*3,row.i*3+3),6*k);arr.set(xyz.subarray(j*3,j*3+3),6*k+3)});
      gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,arr,gl.DYNAMIC_DRAW)
    }
    $("identity").textContent=row.id;$("category").textContent=[row.label,roles[row.i]&1?"visual input":"",roles[row.i]&2?"policy readout":"",roles[row.i]&4?'range input':''].filter(Boolean).join(" · ");
    const links=[...row.in.map(([j,n])=>["in",j,n]),...row.out.map(([j,n])=>["out",j,n])].sort((a,b)=>b[2]-a[2]);
    function showLinks(offset){
      $("neighbors").replaceChildren();const caption=document.createElement("p");caption.textContent=`${fmt(row.in.length)} incoming · ${fmt(row.out.length)} outgoing`;$("neighbors").append(caption);
      for(const [name,j,n] of links.slice(offset,offset+6)){const b=document.createElement("button");b.textContent=`${name} · ${fmt(n)} syn`;b.title="Inspect connected neuron";b.onclick=()=>select("i="+j);$("neighbors").append(b)}
      if(links.length>6){const nav=document.createElement("p");for(const [text,to] of [["‹",offset-6],["›",offset+6]]){const b=document.createElement("button");b.textContent=text;b.disabled=to<0||to>=links.length;b.onclick=()=>showLinks(to);nav.append(b)}nav.append(` ${offset+1}–${Math.min(links.length,offset+6)} of ${fmt(links.length)}`);$("neighbors").append(nav)}
    }
    showLinks(0);
    signal();circuit.select(row.i);sensory?.select(row.i);paint()
  }catch(e){if(ticket===selectionEpoch)$("identity").textContent=e.message}}
  function signal(){if(!sel||!act)return;const v=act[sel.i];$("signal").textContent=`Activation ${v.toFixed(5)} · recent samples`;hist.push(v);if(hist.length>96)hist.shift();
    const c=$("history").getContext("2d");c.clearRect(0,0,250,42);c.beginPath();const amp=Math.max(.02,...hist.map(Math.abs));hist.forEach((n,i)=>{const x=i*250/95,y=21-19*n/amp;i?c.lineTo(x,y):c.moveTo(x,y)});c.strokeStyle="#71eac5";c.stroke()}
  function drawPacket(raw){
    const a=new Uint8Array(raw),len=new DataView(raw).getUint32(0,true);const m=JSON.parse(new TextDecoder().decode(a.subarray(4,4+len)));
    if(m.seq===seq){if(controlState?.state==='paused')$('live').textContent='Ⅱ Paused';else if(Date.now()/1000-m.timestamp>5)$("live").textContent="No new sample";return}let start=4+len;const size=m.side*m.side;
    if(m.count!==meta.nodes||a.length!==start+size+4*m.count)throw Error("Invalid activity snapshot");
    act=new Float32Array(raw.slice(start+size));gl.bindBuffer(gl.ARRAY_BUFFER,activity);gl.bufferSubData(gl.ARRAY_BUFFER,0,act);
    const id=m.action,words=[["","left","right"][id%3],["","down","up"][Math.floor(id/3)%3],id%18>=9?"fire":"",id>=18?"rocket":""].filter(Boolean);
    $("scene").textContent=m.scene;$("frame").textContent=`Frame ${fmt(m.frame)} · Ep ${m.episode}`;$("act").textContent=(words.join(" + ")||"Neutral")+(m.terminal?' · episode ended':'');
    $("metrics").textContent=`${m.device.toUpperCase()} · ${m.inferenceMs} ms inference · ${frame&&m.timestamp>frame.timestamp?((m.seq-seq)/(m.timestamp-frame.timestamp)).toFixed(1):"—"} decisions/s · ${m.drops} publisher drops`;
    $("live").textContent="● Live";frame=m;seq=m.seq;signal();radar.sample(m);
    $('rangeInputOption').hidden=!m.rays; sensory?.sample(m,act);circuit.sample(m,act);
    if(m.training||m.replay){$("live").textContent=m.replay?"► Best replay":`${m.evaluating?"◇ Evaluating":"● "+(m.algorithm||"ppo").toUpperCase()} · update ${m.update}`;$("modelNote").textContent=m.algorithm==="dqn"?`Fixed FlyWire reservoir · Double/dueling DQN · ${m.replaySize??0} replay transitions · epsilon ${(m.epsilon??0).toFixed(3)}. Best model: mean evaluation task reward.`:m.combatRewards?"Fixed full FlyWire reservoir · player-weapon damage/destruction rewards + coverage/RND. Evaluation excludes exploration bonuses; completion remains the main objective.":m.exploration==='episodic-coverage+rnd'?"Fixed full FlyWire reservoir · bounded position coverage + geometry novelty during training. Damage/death earn no exploration bonus; evaluation uses task reward only.":m.rays?"Fixed full FlyWire reservoir · vision + 32-ray inputs. PPO learns the readout with capped geometry novelty during training; evaluation uses task reward only.":"Fixed full FlyWire reservoir. PPO learns the readout; evaluation and best-policy replay use frozen weights."}
    document.body.dataset.seq=String(seq)
  }
  let drag=null;
  atlas.onpointerdown=e=>{if(e.button!==0&&e.button!==1)return;e.preventDefault();atlas.setPointerCapture(e.pointerId);drag=[e.clientX,e.clientY,...pan,yaw,pitch,e.shiftKey||e.button===1]};
  atlas.onauxclick=e=>{if(e.button===1)e.preventDefault()};
  atlas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];
    if(drag[6])pan=[drag[2]+2*dx/atlas.clientWidth/sx,drag[3]+2*dy/atlas.clientHeight/sy];
    else{yaw=drag[4]+dx*.008;pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,drag[5]-dy*.008))}paint()};
  atlas.onpointercancel=()=>{drag=null};
  atlas.onpointerup=e=>{if(!drag)return;const click=!drag[6]&&e.button===0&&Math.hypot(e.clientX-drag[0],e.clientY-drag[1])<4;drag=null;if(!click)return;
    const r=atlas.getBoundingClientRect(),x=(2*(e.clientX-r.left)/r.width-1)/sx-pan[0],y=(2*(e.clientY-r.top)/r.height-1)/sy-pan[1];let best=-1,d=(20/r.width/sx)**2;
    let depth=Infinity;const pixel=(2/r.width/sx)**2;
    for(let i=0;i<xyz.length;i+=3){const a=xyz[i],b=xyz[i+1],c=xyz[i+2];
      const v=(rot[0]*a+rot[3]*b+rot[6]*c-x)**2+(rot[1]*a+rot[4]*b+rot[7]*c-y)**2;
      const z=rot[2]*a+rot[5]*b+rot[8]*c;if(v<d-pixel||(Math.abs(v-d)<=pixel&&z<depth)){d=v;depth=z;best=i/3}}
    if(best>=0)select("i="+best)};
  atlas.onwheel=e=>{e.preventDefault();const z=zoom;zoom=Math.max(.04,Math.min(80,z*Math.exp(-e.deltaY*.0016)));const r=atlas.getBoundingClientRect();pan[0]+=(2*(e.clientX-r.left)/r.width-1)/sx*(z/zoom-1);pan[1]+=(2*(e.clientY-r.top)/r.height-1)/sy*(z/zoom-1);paint()};
  function brainView(){zoom=1;pan=[0,0];yaw=homeYaw;pitch=homePitch;paint()}
  function fitMap(){
    if(!bodyEnabled||!flyBody){brainView();return}
    yaw=.55;pitch=1.05;paint();
    const box=atlas.getBoundingClientRect(),fit=flyBody.fit(rot,box.width/box.height);
    zoom=fit.zoom;pan=fit.pan;paint()
  }
  $("brainView").onclick=brainView;
  $("flyBody").onchange=()=>{bodyEnabled=$("flyBody").checked;$("bodyNote").hidden=!bodyEnabled;fitMap()};
  $("lookup").onsubmit=e=>{e.preventDefault();select("id="+encodeURIComponent($("root").value.trim()))};$("fit").onclick=fitMap;
  $("reset").onclick=()=>{
    selectionEpoch++;sel=null;hist=[];drag=null;
    $("root").value="";$("identity").textContent=emptyInspector;$("category").textContent="";$("signal").textContent="";
    $("neighbors").replaceChildren();const h=$("history");h.getContext("2d").clearRect(0,0,h.width,h.height);
    circuit.select(-1);fitMap()
  };
  for(const id of ["bundled","inputs","outputs","rangeInputs"])$(id).onchange=paint;
  let controlState=null,controlBusy=false,controlToken='';
  function showControl(state){
    controlState=state;if(state.token)controlToken=state.token;
    const ready=state.supported&&!['starting','stopped','unavailable'].includes(state.state);
    $('gamePause').disabled=controlBusy||!ready||state.pauseRequested;
    $('gameStep').disabled=controlBusy||!ready||state.state!=='paused';
    $('gamePlay').disabled=controlBusy||!ready||!state.pauseRequested;
    $('gamePause').setAttribute('aria-pressed',String(state.state==='paused'));
    $('gamePlay').setAttribute('aria-pressed',String(state.state==='playing'&&!state.pauseRequested));
    $('trainingState').textContent=state.error||(!state.supported?'Controller unavailable':
      state.state==='paused'?'Paused · one Step = one model action':
      state.state==='stepping'?'Advancing one action…':state.pauseRequested?'Pausing after current action…':
      state.state==='playing'?'Training playing':state.state==='starting'?'Starting controller…':'Controller stopped');
    if(state.state==='paused')$('live').textContent='Ⅱ Paused';
    document.body.dataset.controlState=state.state;
    document.body.dataset.completedActions=String(state.completedActions??0)
  }
  async function refreshControl(){
    try{showControl(await(await get('/control')).json())}
    catch(e){showControl({state:'unavailable',supported:false,error:e.message})}
    setTimeout(refreshControl,400)
  }
  for(const [id,command] of [['gamePause','pause'],['gameStep','step'],['gamePlay','play']])$(id).onclick=async()=>{
    controlBusy=true;showControl(controlState);
    try{const response=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json','X-NeonFly-Control':controlToken},body:JSON.stringify({command})});
      if(!response.ok)throw Error(await response.text());
      showControl({...controlState,...await response.json()})
    }catch(e){$('trainingState').textContent=e.message}
    finally{controlBusy=false;showControl(controlState)}
  };
  refreshControl();
  new ResizeObserver(paint).observe(atlas);paint();
  loadFlyBody(gl).then(body=>{
    flyBody=body;$("flyBody").disabled=false;
    bodyEnabled=false;
    $("flyBody").checked=bodyEnabled;$("bodyNote").hidden=!bodyEnabled;fitMap();
    document.body.dataset.flyVertices=String(body.info.vertices)
  }).catch(e=>{$("bodyNote").hidden=false;$("bodyNote").textContent=e.message;$("flyBody").disabled=true});
  let progress=null;
  function plot(){
    if(!progress)return;
    const c=$("trend").getContext("2d"),metric=$("metric").value,rows=progress.episodes;
    const finitePoint=([i,v])=>Number.isFinite(i)&&Number.isFinite(v);
    const pts=rows.map(r=>[r.episode,r[metric]??r.reward]).filter(finitePoint);
    const ev=progress.evaluations.map(e=>{
      const seconds=(e.trials||[]).map(t=>t.seconds).filter(Number.isFinite);
      const survival=seconds.length?seconds.reduce((a,b)=>a+b,0)/seconds.length:undefined;
      return [e.episode,metric==="survived_seconds"?survival:e.mean_reward]
    }).filter(finitePoint);
    c.clearRect(0,0,510,208);if(!pts.length&&!ev.length)return;
    const vals=[...pts,...ev].map(p=>p[1]),lo=Math.min(0,...vals),hi=Math.max(lo+.1,...vals),last=Math.max(1,...[...pts,...ev].map(p=>p[0]));
    const X=i=>68+427*i/last,Y=v=>165-(v-lo)*139/(hi-lo);
    c.font="19px Segoe UI";c.fillStyle="#a8b6c4";c.strokeStyle="#344350";
    for(const v of [lo,hi]){c.fillText(v.toFixed(1),4,Y(v)+5);c.beginPath();c.moveTo(65,Y(v));c.lineTo(499,Y(v));c.stroke()}
    c.fillText("0",62,198);c.fillText("Episode",220,198);c.fillText(last,463,198);
    function line(data,color,w){c.strokeStyle=color;c.lineWidth=w;c.beginPath();data.forEach(([i,v],k)=>k?c.lineTo(X(i),Y(v)):c.moveTo(X(i),Y(v)));c.stroke()}
    line(pts,"#758898",1.8);const mean=pts.map(([i],k)=>{const p=pts.slice(Math.max(0,k-9),k+1);return [i,p.reduce((a,b)=>a+b[1],0)/p.length]});line(mean,"#71eac5",3);
    for(const [i,v] of ev){c.fillStyle="#f8ca77";c.beginPath();c.arc(X(i),Y(v),4.5,0,2*Math.PI);c.fill()}
    $("trendNote").textContent=`${progress.run} · ${pts.length} training episodes. Gray: individual · mint: last 10 mean · gold: frozen evaluation.`;
    $("trend").dataset.episodes=String(pts.length)
  }
  $("metric").onchange=plot;
  async function refreshTrend(){try{progress=await(await get("/progress")).json();plot()}
    catch(e){$("trendNote").textContent=e.message}setTimeout(refreshTrend,2500)}
  refreshTrend();
  for(;;){try{const r=await get('/frame');if(r.status===200)drawPacket(await r.arrayBuffer())}catch(e){$("live").textContent="Stream unavailable";$("metrics").textContent=e.message}
    await new Promise(resolve=>setTimeout(resolve,100))}
}
run().catch(e=>{$("live").textContent="Viewer error";$("metrics").textContent=e.message;console.error(e)});
