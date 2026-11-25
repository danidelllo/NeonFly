const names=['Luminance','Red − green','Blue − yellow','Recent motion','Stack motion','Edges'];
const explanations=['Current frame brightness','Red/green color contrast','Blue/yellow color contrast',
  'Brightness change since previous frame ×3','Brightness change across four frames ×2','Local brightness contrast ×4'];
const positive=[[255,255,255],[247,113,113],[106,166,255],[113,234,197],[113,234,197],[113,234,197]];
const negative=[[0,0,0],[97,218,152],[243,211,103],[177,151,255],[177,151,255],[15,23,32]];

export function inputMaps(mapping,project,repaint){
  const $=id=>document.getElementById(id),panel=$('inputMaps'),overlay=$('inputLines'),ctx=overlay.getContext('2d'),hub=$('inputHub');
  let values=null,frame=null,activity=null,bank=-1,pixel=null,selected=-1,source='maps',resource=-1;
  const radar=$('rangePanel');
  const cards=names.map((name,k)=>{
    const b=document.createElement('button');b.className='map-tile';b.type='button';b.setAttribute('aria-pressed','false');
    b.title=explanations[k]+'. Click to trace the whole map. Shift-click a pixel to inspect its neuron mapping.';
    const canvas=document.createElement('canvas');canvas.width=canvas.height=42;canvas.setAttribute('aria-label',name+' 42 by 42 brain input');
    const label=document.createElement('span');label.textContent=name;
    const range=document.createElement('small');range.textContent='42 × 42';b.append(canvas,label,range);panel.append(b);
    b.onclick=e=>{
      source='maps';resource=-1;bank=k;pixel=null;
      if(e.target===canvas&&e.shiftKey){const r=canvas.getBoundingClientRect();
        const x=Math.min(41,Math.max(0,Math.floor((e.clientX-r.left)/r.width*42)));
        const y=Math.min(41,Math.max(0,Math.floor((e.clientY-r.top)/r.height*42)));
        pixel=k*1764+y*42+x;
      }
      refresh();repaint()
    };
    return {button:b,canvas,context:canvas.getContext('2d')}
  });
  const resources=['Health','Bullets','Rockets'].map((name,k)=>{
    const element=document.createElement('button');element.type='button';element.className='resource-input';
    element.setAttribute('aria-pressed','false');element.title=`Trace ${name.toLowerCase()} to its 64 assigned input neurons`;
    element.onclick=()=>{source='resource';resource=k;bank=-1;pixel=null;refresh();repaint()};
    const label=document.createElement('span');label.textContent=name;
    const track=document.createElement('div'),bar=document.createElement('i'),value=document.createElement('b');
    track.append(bar);element.append(label,track,value);$('resourceInputs').append(element);
    return {element,bar,value}
  });
  radar.setAttribute('role','button');radar.tabIndex=0;radar.setAttribute('aria-pressed','false');
  radar.title='Trace the range sensor to its assigned mechanosensory neurons';
  radar.onclick=()=>{source='range';resource=-1;bank=-1;pixel=null;refresh();repaint()};
  radar.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();radar.click()}};
  $('allMaps').onclick=()=>{source='maps';resource=-1;bank=-1;pixel=null;refresh();repaint()};
  $('inputLinks').onchange=repaint;
  $('allInputLinks').onchange=repaint;
  const candidates=Array.from(mapping.visual,(i,k)=>({i,slot:mapping.slots[k]}));
  const bankOf=new Map(candidates.map(d=>[d.i,Math.floor(d.slot/1764)]));
  function refresh(){
    cards.forEach((card,k)=>{card.button.classList.toggle('selected',bank===k);card.button.setAttribute('aria-pressed',String(bank===k))});
    resources.forEach((r,k)=>{const active=source==='resource'&&resource===k;r.element.classList.toggle('selected',active);r.element.setAttribute('aria-pressed',String(active))});
    radar.classList.toggle('selected',source==='range');radar.setAttribute('aria-pressed',String(source==='range'));
    if(!values){$('mapStatus').textContent='Waiting for synchronized encoder maps…';return}
    cards.forEach(({context},k)=>{
      const image=context.createImageData(42,42);let lo=1,hi=-1;
      for(let i=0;i<1764;i++){
        const value=values[k*1764+i];lo=Math.min(lo,value);hi=Math.max(hi,value);
        const magnitude=Math.min(1,Math.abs(value)),target=value<0?negative[k]:positive[k];
        for(let c=0;c<3;c++)image.data[i*4+c]=k===0?Math.round((value+1)*127.5):Math.round(15+(target[c]-15)*magnitude);
        image.data[i*4+3]=255
      }
      context.putImageData(image,0,0);
      if(pixel!==null&&bank===k){context.strokeStyle='#fff';context.lineWidth=.7;context.strokeRect(pixel%42+.1,Math.floor((pixel%1764)/42)+.1,.8,.8)}
      cards[k].button.querySelector('small').textContent=`${lo.toFixed(2)} … ${hi.toFixed(2)}`
    });
    resources.forEach((r,k)=>{const v=frame.resourceDrive[k];r.bar.style.width=`${50*(v+1)}%`;r.value.textContent=v.toFixed(2)});
    if(source==='resource'){
      $('mapStatus').textContent=`${['Health','Bullets','Rockets'][resource]} → 64 input neurons · drive ${frame.resourceDrive[resource].toFixed(4)} · enable Mapping links to trace`;
    }else if(source==='range'){
      $('mapStatus').textContent=`Range sensor → ${mapping.rangeNeurons.length.toLocaleString()} mechanosensory neurons · link colors show neuron activity · enable Mapping links to trace`;
    }else if(pixel!==null){const count=candidates.filter(d=>d.slot===pixel).length;
      $('mapStatus').textContent=`${names[bank]} [${pixel%42}, ${Math.floor((pixel%1764)/42)}] → ${count} input neuron${count===1?'':'s'} · drive ${values[pixel].toFixed(4)}`
    }else if(bank!==-1){const count=candidates.filter(d=>Math.floor(d.slot/1764)===bank).length;
      $('mapStatus').textContent=`${names[bank]} → ${count.toLocaleString()} input neurons · Shift-click a pixel for its exact connections`;
    }else $('mapStatus').textContent=`All inputs · six visual maps + health / bullets / rockets${frame.rays?.valid?' + range sensor':''} · sample ${frame.seq.toLocaleString()}`;
    $('allMaps').setAttribute('aria-pressed',String(source==='maps'&&bank===-1));
    hub.hidden=false
  }
  function sample(m,a){
    frame=m;activity=a;
    if(!m.inputMaps){values=null;refresh();return}
    const description=m.inputMaps;
    if(description.side!==42||description.count!==6||description.encoding!=='float32-le')throw Error('Unsupported input maps');
    const bytes=Uint8Array.from(atob(description.data),c=>c.charCodeAt(0));
    if(bytes.byteLength!==6*42*42*4)throw Error('Incomplete input maps');
    values=new Float32Array(bytes.buffer);refresh()
  }
  function draw(){
    const box=overlay.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,2),w=Math.round(box.width*ratio),h=Math.round(box.height*ratio);
    if(overlay.width!==w||overlay.height!==h){overlay.width=w;overlay.height=h}
    ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,box.width,box.height);
    overlay.dataset.neuronLinks='0';
    if(!values||!$('inputLinks').checked)return;
    const showAll=$('allInputLinks').checked,overview=source==='maps'&&bank===-1;
    const includeRange=!!frame.rays?.valid&&(source==='range'||overview);
    const hb=hub.getBoundingClientRect(),hx=hb.left+hb.width/2-box.left,hy=hb.top+hb.height/2-box.top;
    function line(x,y,X,Y,value,alpha=.45){
      ctx.strokeStyle=value<0?'#b197ff':'#71eac5';ctx.globalAlpha=alpha;ctx.lineWidth=.8;
      ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(X,Y);ctx.stroke()
    }
    if(source==='maps')cards.forEach((card,k)=>{if(bank!==-1&&bank!==k)return;const r=card.canvas.getBoundingClientRect();
      line(r.left+r.width/2-box.left,r.top-box.top,hx,hb.bottom-box.top,1,.23)});
    if(source!=='range'&&bank===-1)resources.forEach((r,k)=>{if(source==='resource'&&resource!==k)return;const rbox=r.element.getBoundingClientRect();line(rbox.left-box.left,rbox.top+rbox.height/2-box.top,hx,hb.bottom-box.top,frame.resourceDrive[k],.45)});
    if(includeRange){const r=radar.getBoundingClientRect();line(r.left+r.width/2-box.left,r.top-box.top,hx,hb.bottom-box.top,1,.45)}
    let targets;
    if(source==='range'){
      targets=frame.rays?.valid?mapping.rangeNeurons.map(i=>({i,value:activity[i]})):[];
      if(!showAll){targets.sort((a,b)=>Math.abs(b.value)-Math.abs(a.value));targets=targets.slice(0,8)}
    }else if(source==='resource')targets=[];
    else if(pixel!==null)targets=candidates.filter(d=>d.slot===pixel);
    else if(bank!==-1){
      targets=candidates.filter(d=>Math.floor(d.slot/1764)===bank);
      if(!showAll)targets.sort((a,b)=>Math.abs(values[b.slot])-Math.abs(values[a.slot])).splice(6)
    }
    else targets=showAll?candidates.slice():names.map((_,k)=>candidates.filter(d=>Math.floor(d.slot/1764)===k)
      .reduce((best,d)=>!best||Math.abs(values[d.slot])>Math.abs(values[best.slot])?d:best,null));
    if(source!=='range'&&bank===-1){
      for(let k=0;k<3;k++){
        if(source==='resource'&&resource!==k)continue;
        const population=mapping.resources.slice(k*64,(k+1)*64);
        const shown=showAll?population:[population.reduce((best,j)=>Math.abs(activity[j])>Math.abs(activity[best])?j:best,population[0])];
        targets.push(...shown.map(i=>({i,value:frame.resourceDrive[k]})))
      }
    }
    if(overview&&includeRange){
      let rangeTargets=mapping.rangeNeurons.map(i=>({i,value:activity[i]}));
      if(!showAll){rangeTargets.sort((a,b)=>Math.abs(b.value)-Math.abs(a.value));rangeTargets=rangeTargets.slice(0,8)}
      targets.push(...rangeTargets)
    }
    const chosen=candidates.find(d=>d.i===selected);
    if(source==='maps'&&chosen&&(bank===-1||Math.floor(chosen.slot/1764)===bank)&&(pixel===null||chosen.slot===pixel)
        &&!targets.some(d=>d.i===selected))targets.push(chosen);
    overlay.dataset.neuronLinks=String(targets.length);
    for(const d of targets){
      const [x,y]=project(d.i,box),v=d.value??values[d.slot];
      const alpha=pixel!==null?.85:d.i===selected?.9:showAll?(overview?.015:.05)+(overview?.045:.1)*Math.abs(v):.18+.36*Math.abs(v);
      line(hx,hy,x,y,v,alpha);
      if(!showAll||pixel!==null||d.i===selected){ctx.fillStyle=v<0?'#b197ff':'#71eac5';ctx.fillRect(x-2,y-2,4,4)}
    }
    ctx.globalAlpha=1
  }
  return {sample,draw,select(i){selected=i;const k=bankOf.get(i);if(source==='maps'&&k!==undefined&&bank!==-1){bank=k;pixel=null;refresh()}}};
}
