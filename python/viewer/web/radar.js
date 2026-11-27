const colors=['#304954','#8bb9ca','#ffc478','#ff7285','#b197ff'];
export function rangeRadar(){
  const panel=document.getElementById('rangePanel'),canvas=document.getElementById('rangeRadar'),ctx=canvas.getContext('2d');
  let available=false,hasScan=false;
  return {sample(frame){
    const rays=frame.rays,active=!!rays?.valid;
    available ||= !!rays;
    panel.hidden=!available;document.querySelector('.field').classList.toggle('has-range',available);
    panel.dataset.live=String(active);
    if(!active){
      if(available)document.getElementById('rangeStatus').textContent=hasScan?'Waiting · last scan shown':'Waiting for range scan';
      return;
    }
    hasScan=true;
    ctx.clearRect(0,0,150,150);ctx.fillStyle='#0b131c';ctx.fillRect(0,0,150,150);
    ctx.strokeStyle='#233742';ctx.lineWidth=.6;
    for(const radius of [20,40,60]){ctx.beginPath();ctx.arc(75,75,radius,0,2*Math.PI);ctx.stroke()}
    ctx.beginPath();ctx.moveTo(10,75);ctx.lineTo(140,75);ctx.moveTo(75,10);ctx.lineTo(75,140);ctx.stroke();
    for(let i=0;i<rays.distance.length;i++){
      const angle=rays.angles?.[i]??2*Math.PI*i/rays.distance.length,distance=rays.distance[i]/rays.range*60;
      const start=(rays.visibleStart?.[i]??0)/rays.range*60;
      const x=75+Math.cos(angle)*distance,y=75-Math.sin(angle)*distance;
      ctx.strokeStyle=colors[rays.kind[i]];ctx.globalAlpha=rays.kind[i]? .85:.3;ctx.lineWidth=rays.kind[i]?1:.6;
      ctx.beginPath();ctx.moveTo(75+Math.cos(angle)*start,75-Math.sin(angle)*start);ctx.lineTo(x,y);ctx.stroke();
      if(rays.kind[i]){ctx.fillStyle=colors[rays.kind[i]];ctx.fillRect(x-1.5,y-1.5,3,3)}
    }
    ctx.globalAlpha=1;ctx.fillStyle='#dceef0';ctx.beginPath();ctx.moveTo(80,75);ctx.lineTo(71,71);ctx.lineTo(71,79);ctx.closePath();ctx.fill();
    ctx.fillStyle='#7796a5';ctx.font='8px Segoe UI';ctx.fillText('↑',72,9);ctx.fillText('→',140,78);
    const visible=rays.visible.some((end,i)=>end>(rays.visibleStart?.[i]??0));
    document.getElementById('rangeStatus').textContent=rays.insideView?`${rays.distance.length} rays · ${rays.range} units · visible scene`:
      visible?`${rays.distance.length} rays · outside ship · visible scene`:`${rays.distance.length} rays · no visible segments`;
    canvas.title=`Same decision sample ${frame.seq}. ${rays.distance.length===126?'Front 55 · Top 8 · Behind 8 · Below 55. ':''}Right = forward, up = world up. Solid: blue; damageable: amber; hostile projectile: red; lethal: violet. Clipped to current camera view.`;
    panel.dataset.rayCount=String(rays.distance.length);
    panel.dataset.sample=String(frame.seq)
  }};
}
