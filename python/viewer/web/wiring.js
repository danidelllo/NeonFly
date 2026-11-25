const $=id=>document.getElementById(id);
const controls=[["Left",1],["Right",2],["Down",3],["Up",6],["Primary fire",9],["Rocket",18]];
function decoded(a){return [a%3===1,a%3===2,Math.floor(a/3)%3===1,Math.floor(a/3)%3===2,Math.floor(a/9)%2===1,a>=18]}
const label=a=>controls.filter((_,i)=>decoded(a)[i]).map(c=>c[0]).join(" + ")||"Neutral";
const signed=n=>(n>=0?"+":"")+n.toFixed(4);
export function actionWiring(outputs,project,repaint){
  const combo=$("combo"),canvas=$("wiring"),ctx=canvas.getContext("2d"),hub=$("hub");
  let frame,act,weights,key,pending=false,retry=0,selected=-1,drivers=[],action=0;
  const buttons=controls.map(([name,solo])=>{
    const b=document.createElement("button");b.textContent=name;b.title="Inspect "+name+" alone (does not control game)";
    b.onclick=()=>{combo.value=String(solo);refresh()};$("keys").append(b);return b
  });
  combo.add(new Option("Follow live choice","live"));
  for(let i=0;i<36;i++)combo.add(new Option(`#${i} · ${label(i)}`,String(i)));
  combo.onchange=refresh;$("allLinks").onchange=refresh;$("readoutLinks").onchange=repaint;
  function refresh(){
    if(!frame)return;
    action=combo.value==="live"?frame.action:Number(combo.value);
    const live=decoded(frame.action);buttons.forEach((b,i)=>b.classList.toggle("on",live[i]));
    drivers=[];hub.hidden=!(weights||frame.readout==="nonlinear");
    if(weights){
      const width=outputs.length,offset=action*(width+1),normalized=frame.training||frame.replay;
      let mean=0,variance=0;
      if(normalized){
        for(const i of outputs)mean+=act[i];mean/=width;
        for(const i of outputs)variance+=(act[i]-mean)**2;variance/=width
      }
      const scale=normalized?1/Math.sqrt(variance+1e-5):1,bias=weights[offset+width];
      let score=bias;
      for(let j=0;j<width;j++){
        const w=weights[offset+j],v=(act[outputs[j]]-mean)*scale*w;
        score+=v;drivers.push({i:outputs[j],v,w})
      }
      drivers.sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));
      const inspected=drivers.find(d=>d.i===selected);
      if(!$("allLinks").checked){drivers=drivers.slice(0,8);if(inspected&&!drivers.includes(inspected))drivers.push(inspected)}
      hub.textContent="#"+action;
      $("score").textContent=`#${action} ${label(action)}\nScore ${signed(score)} · bias ${signed(bias)}\n${drivers.length} of ${width} inputs shown`;
      const d=inspected||drivers[0];
      $("driver").textContent=`${inspected?"Selected":"Strongest"} neuron [${d.i}]\nWeight ${signed(d.w)} · term ${signed(d.v)}`;
    }else if(frame.readout==="nonlinear"){
      drivers=Array.from(outputs,i=>({i,v:act[i],w:null})).sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));
      const inspected=drivers.find(d=>d.i===selected);
      if(!$("allLinks").checked){drivers=drivers.slice(0,8);if(inspected&&!drivers.includes(inspected))drivers.push(inspected)}
      hub.textContent=`#${action}`;
      hub.title="Layer-normalized activity, differences and averages from 1,415 output neurons form 4,245 readout features. Lines show live activity, not linear weight contributions.";
      $("score").textContent=`#${action} ${label(action)}\nScore ${signed(frame.logits[action])}\nHidden layers 256 → 128 · 36 action scores`;
      $("driver").textContent=`${drivers.length} of ${outputs.length} neuron links shown, ranked by activity magnitude. Lines show wiring and activity, not causal contribution.`;
    }else{$("driver").textContent=""}
    repaint()
  }
  function sample(m,a){
    frame=m;act=a;
    const wanted=JSON.stringify([m.run||"",m.training?"train":m.replay?"replay":"demo",m.update??null]);
    if(key!==wanted){weights=null;key=wanted;retry=0;$("score").textContent="Waiting for matching policy weights…"}
    if(frame.readout!=="nonlinear"&&!weights&&!pending&&Date.now()>=retry){
      pending=true;
      (async()=>{try{
        const r=await fetch("/actor",{cache:"no-store"});if(!r.ok)throw Error(await r.text());
        const raw=await r.arrayBuffer(),n=new DataView(raw).getUint32(0,true);
        const received=JSON.stringify(JSON.parse(new TextDecoder().decode(new Uint8Array(raw,4,n))));
        if(received!==key)throw Error("Waiting for actor matching this frame");
        const w=new Float32Array(raw.slice(4+n));if(w.length!==36*(outputs.length+1))throw Error("Unexpected policy dimensions");
        weights=w;refresh()
      }catch(e){$("score").textContent=e.message;retry=Date.now()+1500}finally{pending=false}})()
    }
    refresh()
  }
  function draw(){
    const box=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,2);
    const w=Math.round(box.width*ratio),h=Math.round(box.height*ratio);if(!w||!h)return;
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
    ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,box.width,box.height);
    if(!drivers.length||!$("readoutLinks").checked)return;
    const hb=hub.getBoundingClientRect(),x=hb.left-box.left,y=hb.top+hb.height/2-box.top;
    const max=Math.abs(drivers[0].v)||1;
    for(const d of drivers){
      const [px,py]=project(d.i,box.width,box.height),strength=Math.abs(d.v)/max;
      ctx.strokeStyle=d.v<0?"#b197ff":"#71eac5";ctx.globalAlpha=d.i===selected?1:.06+.68*strength;
      ctx.lineWidth=d.i===selected?2.6:.55+strength;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(x,y);ctx.stroke();
      if(drivers.length<12||d.i===selected){ctx.fillStyle=ctx.strokeStyle;ctx.fillRect(px-2,py-2,4,4)}
    }
    ctx.globalAlpha=1;ctx.lineWidth=1.5;ctx.strokeStyle="#ecc073";
    decoded(action).forEach((on,i)=>{if(on){const b=buttons[i].getBoundingClientRect();ctx.beginPath();ctx.moveTo(x+hb.width,y);ctx.lineTo(b.left-box.left,b.top+b.height/2-box.top);ctx.stroke()}})
  }
  return {draw,sample,select(i){selected=i;refresh()}}
}
