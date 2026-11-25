// Static anatomical context only: shares the atlas camera, never touches the agent.
export async function loadFlyBody(gl){
  const [infoResponse,meshResponse]=await Promise.all([fetch('/assets/fly/model.json'),fetch('/assets/fly/body.bin')]);
  if(!infoResponse.ok||!meshResponse.ok)throw Error('Fly body asset unavailable');
  const info=await infoResponse.json(),raw=await meshResponse.arrayBuffer();
  if(raw.byteLength!==info.vertices*6*4)throw Error('Fly body mesh is incomplete');
  function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
  const program=gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,`#version 300 es
    layout(location=0) in vec3 position;layout(location=1) in vec3 normal;
    uniform mat3 rotation;uniform vec2 scale,pan;uniform float depth;
    out vec3 n;
    void main(){vec3 p=rotation*position;n=rotation*normal;
      gl_Position=vec4((p.xy+pan)*scale,p.z*depth,1.);}`));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`#version 300 es
    precision highp float;in vec3 n;out vec4 color;
    void main(){vec3 normal=normalize(n);
      float rim=pow(1.-abs(normal.z),3.);
      float light=.5+.5*abs(dot(normal,normalize(vec3(-.4,-.6,1.))));
      color=vec4(mix(vec3(.22,.40,.49),vec3(.44,.79,.76),rim)*light,.012+.085*rim);}`));
  gl.linkProgram(program);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
  const u=Object.fromEntries(['rotation','scale','pan','depth'].map(k=>[k,gl.getUniformLocation(program,k)]));
  const vao=gl.createVertexArray(),buffer=gl.createBuffer();
  gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(raw),gl.STATIC_DRAW);
  for(let i=0;i<2;i++){gl.enableVertexAttribArray(i);gl.vertexAttribPointer(i,3,gl.FLOAT,false,24,i*12)}
  gl.bindVertexArray(null);
  const radius=Math.max(...info.bounds.flat().map(Math.abs))*1.8;
  return {info,depth:.9/radius,
    draw(rotation,sx,sy,pan){
      gl.useProgram(program);gl.bindVertexArray(vao);gl.disable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.depthMask(false);
      gl.uniformMatrix3fv(u.rotation,false,rotation);gl.uniform2f(u.scale,sx,-sy);gl.uniform2fv(u.pan,pan);gl.uniform1f(u.depth,.9/radius);
      gl.drawArrays(gl.TRIANGLES,0,info.vertices);gl.depthMask(true);gl.bindVertexArray(null);
    },
    fit(rotation,aspect){
      // Fit the actual posed surface, not the mostly empty corners of its box.
      const vertices=new Float32Array(raw),lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];
      for(let i=0;i<vertices.length;i+=6){
        const x=vertices[i],y=vertices[i+1],z=vertices[i+2];
        const q=[rotation[0]*x+rotation[3]*y+rotation[6]*z,
                 rotation[1]*x+rotation[4]*y+rotation[7]*z];
        for(let k=0;k<2;k++){lo[k]=Math.min(lo[k],q[k]);hi[k]=Math.max(hi[k],q[k])}
      }
      const bx=Math.min(1,1.55/aspect),by=bx*aspect;
      return {zoom:Math.min(1.6/((hi[0]-lo[0])*bx),1.65/((hi[1]-lo[1])*by)),pan:lo.map((v,k)=>-(v+hi[k])/2)};
    }
  };
}
