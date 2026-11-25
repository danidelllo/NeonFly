const colors=['#304954','#8bb9ca','#ffc478','#ff7285','#b197ff'];
export function rangeRadar(){
  const panel=document.getElementById('rangePanel'),canvas=document.getElementById('rangeRadar'),ctx=canvas.getContext('2d');
  return {sample(frame){
    const rays=frame.rays,active=!!rays?.valid;
    panel.hidden=!active;document.querySelector('.field').classList.toggle('has-range',active);
    if(!active)return;
    ctx.clearRect(0,0,150,150);ctx.fillStyle='#0b131c';ctx.fillRect(0,0,150,150);
    ctx.strokeStyle='#233742';ctx.lineWidth=.6;
    for(const radius of [20,40,60]){ctx.beginPath();ctx.arc(75,75,radius,0,2*Math.PI);ctx.stroke()}
    ctx.beginPath();ctx.moveTo(10,75);ctx.lineTo(140,75);ctx.moveTo(75,10);ctx.lineTo(75,140);ctx.stroke();
    for(let i=0;i<32;i++){
      const angle=2*Math.PI*i/32,distance=rays.distance[i]/rays.range*60;
      const start=(rays.visibleStart?.[i]??0)/rays.range*60;
      const x=75+Math.cos(angle)*distance,y=75-Math.sin(angle)*distance;
      ctx.strokeStyle=colors[rays.kind[i]];ctx.globalAlpha=rays.kind[i]? .85:.3;ctx.lineWidth=rays.kind[i]?1:.6;
      ctx.beginPath();ctx.moveTo(75+Math.cos(angle)*start,75-Math.sin(angle)*start);ctx.lineTo(x,y);ctx.stroke();
      if(rays.kind[i]){ctx.fillStyle=colors[rays.kind[i]];ctx.fillRect(x-1.5,y-1.5,3,3)}
    }
    ctx.globalAlpha=1;ctx.fillStyle='#dceef0';ctx.beginPath();ctx.moveTo(80,75);ctx.lineTo(71,71);ctx.lineTo(71,79);ctx.closePath();ctx.fill();
    ctx.fillStyle='#7796a5';ctx.font='8px Segoe UI';ctx.fillText('↑',72,9);ctx.fillText('→',140,78);
    const visible=rays.visible.some((end,i)=>end>(rays.visibleStart?.[i]??0));
    document.getElementById('rangeStatus').textContent=rays.insideView?`32 rays · ${rays.range} units · visible scene`:
      visible?'Ship outside · sensing visible scene':'Ship outside · no visible ray segments';
    canvas.title=`Same decision sample ${frame.seq}. Right = forward, up = world up. Solid: blue; damageable: amber; hostile projectile: red; lethal: violet. Clipped to current camera view.`;
    panel.dataset.sample=String(frame.seq)
  }};
}
