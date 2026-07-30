import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import {
  Layers, Box, Activity, Waves, Magnet, Radio, Gauge, Database, Upload,
  Trash2, GitCompare, ChevronDown, ChevronUp, Loader2, Save, History,
  FolderOpen, Download, Filter, Map, BarChart2, Play, Maximize2, Minimize2,
  Eye, EyeOff, Settings, RefreshCw, AlertCircle, CheckCircle, Info,
  Grid, Sliders, Mountain, FileText
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, AreaChart, Area, BarChart, Bar
} from 'recharts';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { MarchingCubes } from 'three/examples/jsm/objects/MarchingCubes.js';

// ── useTheme hook: tema değişince tüm uygulamayı yeniden render et ────────────
function useTheme() {
  const [theme, setThemeState] = React.useState(_theme);
  const toggle = React.useCallback((t) => {
    const next = t || (_theme === 'dark' ? 'light' : 'dark');
    applyTheme(next);
    setThemeState(next);
  }, []);
  return [theme, toggle];
}

// ─── Renk paleti ─────────────────────────────────────────────────────────────
// ── Tema sistemi ─────────────────────────────────────────────────────────────
const THEMES = {
  dark: {
    bg:       '#0A0C0F',
    surface:  '#141720',
    panel:    '#1A1F2E',
    border:   '#252B3B',
    borderStr:'#323A50',
    header:   '#E5E8EE',
    accent:   '#E8A020',
    accentL:  '#F2BC55',
    teal:     '#2AABCC',
    tealL:    '#5CC8E0',
    text:     '#E5E8EE',
    textMid:  '#8A94A8',
    textLow:  '#4E566A',
    ok:       '#27A865',
    warn:     '#E8A020',
    err:      '#E03A3A',
    purple:   '#8B5CF6',
  },
  light: {
    bg:       '#F2F4F7',
    surface:  '#FFFFFF',
    panel:    '#EAECF0',
    border:   '#CDD2DC',
    borderStr:'#A8B0C0',
    header:   '#0D1117',
    accent:   '#C47A10',
    accentL:  '#E09030',
    teal:     '#1A7A96',
    tealL:    '#3A9AB8',
    text:     '#0D1117',
    textMid:  '#3A4252',
    textLow:  '#6B7488',
    ok:       '#1A8A52',
    warn:     '#B86A00',
    err:      '#B82020',
    purple:   '#6B3FBE',
  },
};

// Global tema state — modülün en üstünde tutuluyor, React state değil
// çünkü C objesi bileşen dışında da kullanılıyor
let _theme = 'dark';
let C = { ...THEMES.dark };

function applyTheme(t) {
  _theme = t;
  Object.assign(C, THEMES[t]);
  // localStorage'a kaydet
  try { localStorage.setItem('geopinn_theme', t); } catch(e){}
}

// Başlangıçta kayıtlı temayı yükle
try {
  const saved = localStorage.getItem('geopinn_theme');
  if (saved === 'light' || saved === 'dark') applyTheme(saved);
} catch(e){}

// ─── Colormap (Viridis benzeri, jeoloji için) ─────────────────────────────────
function scalarToColor(t) {
  const stops = [
    [0.00, [68,  1, 84]],
    [0.20, [59, 82,139]],
    [0.40, [33,145,140]],
    [0.60, [94,201, 98]],
    [0.80, [253,231, 37]],
    [1.00, [255,255,255]],
  ];
  let lo = stops[0], hi = stops[stops.length-1];
  for (let i = 0; i < stops.length-1; i++) {
    if (t >= stops[i][0] && t <= stops[i+1][0]) { lo=stops[i]; hi=stops[i+1]; break; }
  }
  const f = (t - lo[0]) / (hi[0] - lo[0] + 1e-9);
  return lo[1].map((c,i) => (c + f*(hi[1][i]-c))/255);
}
function scalarToCSSColor(t) {
  const [r,g,b] = scalarToColor(t).map(c => Math.round(c*255));
  return `rgb(${r},${g},${b})`;
}

// ─── Yardımcılar ──────────────────────────────────────────────────────────────
function percentile(arr, p) {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a,b)=>a-b);
  return sorted[Math.min(sorted.length-1, Math.floor(sorted.length*p))];
}

function trilinearUpsample(data, factor) {
  const n = data.length, m = n*factor, out = [];
  for (let i=0;i<m;i++) {
    const gi=i/factor, i0=Math.min(Math.floor(gi),n-1), i1=Math.min(i0+1,n-1), fi=gi-i0;
    out.push([]);
    for (let j=0;j<m;j++) {
      const gj=j/factor, j0=Math.min(Math.floor(gj),n-1), j1=Math.min(j0+1,n-1), fj=gj-j0;
      out[i].push([]);
      for (let k=0;k<m;k++) {
        const gk=k/factor, k0=Math.min(Math.floor(gk),n-1), k1=Math.min(k0+1,n-1), fk=gk-k0;
        const c000=data[i0][j0][k0],c100=data[i1][j0][k0],c010=data[i0][j1][k0],c110=data[i1][j1][k0];
        const c001=data[i0][j0][k1],c101=data[i1][j0][k1],c011=data[i0][j1][k1],c111=data[i1][j1][k1];
        const c00=c000*(1-fi)+c100*fi, c10=c010*(1-fi)+c110*fi;
        const c01=c001*(1-fi)+c101*fi, c11=c011*(1-fi)+c111*fi;
        const c0=c00*(1-fj)+c10*fj, c1=c01*(1-fj)+c11*fj;
        out[i][j].push(c0*(1-fk)+c1*fk);
      }
    }
  }
  return out;
}

// ─── UI bileşenleri ───────────────────────────────────────────────────────────
function PanelSection({ title, icon: Icon, children, defaultOpen = true, badge }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ borderBottom: `1px solid ${C.border}` }}>
      <button onClick={() => setOpen(o=>!o)}
        style={{ width:'100%', display:'flex', alignItems:'center', gap:6, padding:'8px 12px',
                 background:'none', border:'none', cursor:'pointer', color: C.header,
                 fontSize:11, fontWeight:700, letterSpacing:'0.06em', textTransform:'uppercase' }}>
        {Icon && <Icon size={12} />}
        <span style={{ flex:1, textAlign:'left' }}>{title}</span>
        {badge && (
          <span style={{ background: C.accent, color:'#fff', borderRadius:10,
                         padding:'1px 6px', fontSize:9, fontWeight:700 }}>{badge}</span>
        )}
        {open ? <ChevronUp size={12}/> : <ChevronDown size={12}/>}
      </button>
      {open && <div style={{ padding:'0 12px 12px' }}>{children}</div>}
    </div>
  );
}

function MetricCard({ label, value, unit, icon: Icon, color = C.teal, delta }) {
  return (
    <div style={{ background: C.surface, border:`1px solid ${C.border}`, borderRadius:6,
                  padding:'10px 12px', display:'flex', alignItems:'center', gap:10 }}>
      <div style={{ width:32, height:32, borderRadius:8, background:`${color}18`,
                    display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0 }}>
        {Icon && <Icon size={16} color={color} />}
      </div>
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ fontSize:10, color: C.textLow, textTransform:'uppercase', letterSpacing:'0.05em' }}>{label}</div>
        <div style={{ fontSize:18, fontWeight:700, color: C.text, fontFamily:'monospace', lineHeight:1.2 }}>
          {value}<span style={{ fontSize:10, color: C.textMid, marginLeft:3, fontFamily:'inherit' }}>{unit}</span>
        </div>
        {delta !== undefined && (
          <div style={{ fontSize:9, color: delta >= 0 ? C.ok : C.err, marginTop:1 }}>
            {delta >= 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(3)}
          </div>
        )}
      </div>
    </div>
  );
}

function Btn({ children, onClick, disabled, variant = 'primary', size = 'md', icon: Icon, style: extraStyle }) {
  const base = {
    display:'flex', alignItems:'center', justifyContent:'center', gap:6,
    border:'none', cursor: disabled ? 'not-allowed' : 'pointer',
    fontWeight:600, borderRadius:5, transition:'all 0.15s',
    opacity: disabled ? 0.5 : 1,
    fontSize: size === 'sm' ? 11 : 13,
    padding: size === 'sm' ? '5px 10px' : '8px 16px',
    ...extraStyle,
  };
  const variants = {
    primary: { background: C.accent, color:'#fff' },
    secondary: { background: C.surface, color: C.header, border:`1px solid ${C.border}` },
    danger: { background:'#FEF2F2', color: C.err, border:`1px solid ${C.err}40` },
    teal: { background: C.teal, color:'#fff' },
    ghost: { background:'transparent', color: C.textMid, border:`1px solid transparent` },
  };
  return (
    <button onClick={onClick} disabled={disabled} style={{ ...base, ...variants[variant] }}>
      {Icon && <Icon size={size === 'sm' ? 12 : 14} />}
      {children}
    </button>
  );
}

function Select({ value, onChange, options, style }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)}
      style={{ width:'100%', padding:'5px 8px', border:`1px solid ${C.border}`,
               borderRadius:5, background: C.surface, color: C.text, fontSize:12,
               fontFamily:'inherit', cursor:'pointer', ...style }}>
      {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

function RangeRow({ label, value, min, max, step = 1, onChange, format = v => v }) {
  return (
    <div style={{ marginBottom:8 }}>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:3 }}>
        <span style={{ fontSize:11, color: C.textMid }}>{label}</span>
        <span style={{ fontSize:11, fontFamily:'monospace', color: C.teal, fontWeight:700 }}>{format(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width:'100%', accentColor: C.teal }} />
    </div>
  );
}

function Tag({ children, color = C.teal }) {
  return (
    <span style={{ background:`${color}18`, color, border:`1px solid ${color}40`,
                   borderRadius:4, padding:'2px 7px', fontSize:10, fontWeight:600 }}>
      {children}
    </span>
  );
}

// ─── Colorbar bileşeni ────────────────────────────────────────────────────────
function ColorBar({ vmin = 0, vmax = 1, label = 'Değer', horizontal = false }) {
  const stops = [0,0.2,0.4,0.6,0.8,1].map(t => `${scalarToCSSColor(t)} ${t*100}%`).join(',');
  const grad = horizontal
    ? `linear-gradient(to right, ${stops})`
    : `linear-gradient(to top, ${stops})`;
  return (
    <div style={{ position:'absolute', bottom:16, right:16, zIndex:20,
                  display:'flex', flexDirection:'column', alignItems:'center', gap:4,
                  background:'rgba(255,255,255,0.92)', border:`1px solid ${C.border}`,
                  borderRadius:6, padding:'8px 10px', boxShadow:'0 2px 8px rgba(0,0,0,0.12)' }}>
      <span style={{ fontSize:9, color: C.textMid, textTransform:'uppercase', letterSpacing:'0.06em', fontWeight:700 }}>{label}</span>
      <div style={{ display:'flex', gap:6, alignItems:'stretch' }}>
        <div style={{ display:'flex', flexDirection:'column', justifyContent:'space-between',
                      fontSize:9, fontFamily:'monospace', color: C.textMid, textAlign:'right' }}>
          <span>{vmax.toFixed(3)}</span>
          <span>{((vmin+vmax)/2).toFixed(3)}</span>
          <span>{vmin.toFixed(3)}</span>
        </div>
        <div style={{ width:14, height:100, background:grad, borderRadius:3,
                      border:`1px solid ${C.border}` }} />
      </div>
    </div>
  );
}

// ─── 3D Izoyüzey ─────────────────────────────────────────────────────────────
function IsosurfaceMesh({ modelData, isoThreshold, opacity }) {
  const res = 32;
  const { mc } = useMemo(() => {
    const mat = new THREE.MeshStandardMaterial({
      vertexColors:true, roughness:0.35, metalness:0.08,
      transparent: opacity < 1, opacity,
    });
    const mc = new MarchingCubes(res, mat, true, true, 120000);
    mc.scale.set(8,8,8);
    return { mc };
  }, []);

  useEffect(() => { mc.material.opacity = opacity; mc.material.transparent = opacity < 1; },
    [opacity, mc]);

  useEffect(() => {
    if (!modelData?.length) return;
    const factor = Math.max(1, Math.round(res / modelData.length));
    const fine = trilinearUpsample(modelData, factor);
    const n = fine.length, sz = res, field = mc.field;
    field.fill(0);
    const flat = [];
    for (let x=0;x<sz;x++) {
      const fx=Math.min(Math.floor(x/sz*n),n-1);
      for (let y=0;y<sz;y++) {
        const fy=Math.min(Math.floor(y/sz*n),n-1);
        for (let z=0;z<sz;z++) {
          const fz=Math.min(Math.floor(z/sz*n),n-1);
          const v=fine[fx][fy][fz];
          field[x+y*sz+z*sz*sz]=v; flat.push(v);
        }
      }
    }
    const vmin=percentile(flat,0.02), vmax=percentile(flat,0.98);
    mc.isolation = percentile(flat, 1-isoThreshold);
    mc.update();
    const geo=mc.geometry;
    if (geo?.attributes?.position) {
      const pos=geo.attributes.position, cnt=pos.count;
      const cols=new Float32Array(cnt*3);
      for (let i=0;i<cnt;i++) {
        const gx=Math.min(sz-1,Math.max(0,Math.round(((pos.getX(i)/8)*0.5+0.5)*sz)));
        const gy=Math.min(sz-1,Math.max(0,Math.round(((pos.getY(i)/8)*0.5+0.5)*sz)));
        const gz=Math.min(sz-1,Math.max(0,Math.round(((pos.getZ(i)/8)*0.5+0.5)*sz)));
        const v=field[gx+gy*sz+gz*sz*sz];
        const t=Math.max(0,Math.min(1,(v-vmin)/(vmax-vmin+1e-9)));
        const [r,g,b]=scalarToColor(t);
        cols[i*3]=r; cols[i*3+1]=g; cols[i*3+2]=b;
      }
      geo.setAttribute('color',new THREE.BufferAttribute(cols,3));
    }
  }, [modelData, mc, isoThreshold]);

  return <primitive object={mc}/>;
}

// Topografik düzlem
function TopographyPlane({ show, opacity = 0.6 }) {
  if (!show) return null;
  const geo = useMemo(() => {
    const g = new THREE.PlaneGeometry(16, 16, 31, 31);
    const pos = g.attributes.position;
    for (let i=0;i<pos.count;i++) {
      const x=pos.getX(i)/8, y=pos.getY(i)/8;
      const h = 0.5*Math.sin(x*3)*Math.cos(y*2) + 0.3*Math.cos(x*5+y*3);
      pos.setZ(i, h*1.5);
    }
    g.computeVertexNormals();
    return g;
  }, []);

  return (
    <mesh geometry={geo} position={[0, 12, 0]} rotation={[-Math.PI/2, 0, 0]}>
      <meshStandardMaterial color="#5B7A4E" transparent opacity={opacity}
        wireframe={false} side={THREE.DoubleSide} />
    </mesh>
  );
}

function Scene3D({ modelData, isoThreshold, opacity, showTopo, topoOpacity, bgColor }) {
  return (
    <Canvas camera={{ position:[22,18,22], fov:45 }}
      style={{ background: bgColor === 'dark' ? '#0F172A' : '#EFF6FF' }}>
      <ambientLight intensity={0.9} />
      <directionalLight position={[15,20,10]} intensity={1.3} castShadow />
      <directionalLight position={[-15,-10,-10]} intensity={0.4} />
      <pointLight position={[0,25,0]} intensity={0.3} color="#E0F2FE" />
      {modelData?.length > 0 && (
        <IsosurfaceMesh modelData={modelData} isoThreshold={isoThreshold} opacity={opacity} />
      )}
      <TopographyPlane show={showTopo} opacity={topoOpacity} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.06} />
      <gridHelper args={[40,20, bgColor==='dark'?'#1E293B':'#CBD5E1', bgColor==='dark'?'#0F172A':'#E2E8F0']}
        position={[0,-12,0]} />
      {/* Eksen göstergesi */}
      <axesHelper args={[6]} position={[-9,-11,-9]} />
    </Canvas>
  );
}

// ─── Slice View ───────────────────────────────────────────────────────────────
function SliceView({ modelData, axis, idx, colorRange, sweeping }) {
  const containerRef=useRef(), canvasRef=useRef(), offRef=useRef();
  const stRef=useRef({ zoom:1, panX:0, panY:0, drag:false, lx:0, ly:0 });
  const [tip, setTip]=useState(null);

  const render = useCallback(() => {
    const cv=canvasRef.current, off=offRef.current;
    if (!cv||!off) return;
    const {zoom,panX,panY}=stRef.current;
    const W=cv.width, H=cv.height;
    const ctx=cv.getContext('2d');
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='#EFF6FF'; ctx.fillRect(0,0,W,H);
    const side=off.width*zoom;
    const x0=(W-side)/2+panX, y0=(H-side)/2+panY;
    ctx.imageSmoothingEnabled=zoom<3;
    ctx.drawImage(off,x0,y0,side,side);
    // ızgara
    if (zoom>6&&off.width<=64) {
      const cell=zoom;
      ctx.strokeStyle='rgba(0,0,0,0.08)'; ctx.lineWidth=0.5;
      for (let g=((x0%cell)+cell)%cell;g<=W;g+=cell) {
        ctx.beginPath();ctx.moveTo(g,0);ctx.lineTo(g,H);ctx.stroke();
      }
      for (let g=((y0%cell)+cell)%cell;g<=H;g+=cell) {
        ctx.beginPath();ctx.moveTo(0,g);ctx.lineTo(W,g);ctx.stroke();
      }
    }
    // Kesit çizgisi
    ctx.strokeStyle=`${C.accent}`; ctx.lineWidth=1.5; ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();
    ctx.setLineDash([]);
    // Koordinat göstergesi
    ctx.fillStyle=C.header; ctx.font='bold 10px monospace';
    ctx.fillText(`${axis.toUpperCase()} = ${idx}`, 8, 16);
  }, [axis, idx]);

  useEffect(() => {
    if (!modelData?.length) return;
    const n=modelData.length, i=Math.min(idx,n-1);
    const vmin=colorRange?.min??0, vmax=colorRange?.max??1;
    const off=document.createElement('canvas');
    off.width=n; off.height=n;
    const ctx=off.getContext('2d');
    const img=ctx.createImageData(n,n);
    for (let row=0;row<n;row++) {
      for (let col=0;col<n;col++) {
        const ri=n-1-row;
        let v;
        if (axis==='z')      v=modelData[col]?.[ri]?.[i]??0;
        else if (axis==='y') v=modelData[col]?.[i]?.[ri]??0;
        else                 v=modelData[i]?.[col]?.[ri]??0;
        const t=Math.max(0,Math.min(1,(v-vmin)/(vmax-vmin+1e-9)));
        const [r,g,b]=scalarToColor(t).map(c=>Math.round(c*255));
        const px=(row*n+col)*4;
        img.data[px]=r;img.data[px+1]=g;img.data[px+2]=b;img.data[px+3]=255;
      }
    }
    ctx.putImageData(img,0,0);
    offRef.current=off;
    const cv=canvasRef.current;
    const sc=cv?Math.min(cv.width,cv.height)/n:1;
    stRef.current={zoom:sc,panX:0,panY:0,drag:false,lx:0,ly:0};
    render();
  },[modelData,axis,idx,colorRange,render]);

  useEffect(() => {
    const el=containerRef.current, cv=canvasRef.current;
    if (!el||!cv) return;
    const ro=new ResizeObserver(()=>{cv.width=el.clientWidth;cv.height=el.clientHeight;render();});
    ro.observe(el);
    return ()=>ro.disconnect();
  },[render]);

  const onWheel=e=>{
    e.preventDefault();
    const s=stRef.current;
    s.zoom=Math.max(0.5,Math.min(30,s.zoom*(e.deltaY<0?1.15:1/1.15)));
    render();
  };
  const onDown=e=>{const s=stRef.current;s.drag=true;s.lx=e.clientX;s.ly=e.clientY;};
  const onMove=e=>{
    const s=stRef.current, cv=canvasRef.current, off=offRef.current;
    if (s.drag){s.panX+=e.clientX-s.lx;s.panY+=e.clientY-s.ly;s.lx=e.clientX;s.ly=e.clientY;render();}
    if (!cv||!off) return;
    const {zoom,panX,panY}=s;
    const rect=cv.getBoundingClientRect();
    const mx=(e.clientX-rect.left)*(cv.width/rect.width);
    const my=(e.clientY-rect.top)*(cv.height/rect.height);
    const side=off.width*zoom;
    const x0=(cv.width-side)/2+panX, y0=(cv.height-side)/2+panY;
    const gx=Math.floor((mx-x0)/zoom), gy=Math.floor((my-y0)/zoom);
    if (gx>=0&&gx<off.width&&gy>=0&&gy<off.height) {
      const n=modelData.length, si=Math.min(idx,n-1);
      const ri=n-1-gy;
      let v;
      if (axis==='z') v=modelData[gx]?.[ri]?.[si];
      else if (axis==='y') v=modelData[gx]?.[si]?.[ri];
      else v=modelData[si]?.[gx]?.[ri];
      setTip({x:e.clientX-rect.left,y:e.clientY-rect.top,gx,gy:ri,v:v?.toFixed(4)??'-'});
    } else setTip(null);
  };
  const onUp=()=>{stRef.current.drag=false;};
  const onDbl=()=>{
    const cv=canvasRef.current,off=offRef.current;
    const sc=(cv&&off)?Math.min(cv.width,cv.height)/off.width:1;
    stRef.current={...stRef.current,zoom:sc,panX:0,panY:0};render();
  };

  return (
    <div ref={containerRef} style={{ width:'100%',height:'100%',position:'relative',
                                     overflow:'hidden',cursor:'crosshair',background:'#EFF6FF' }}
      onWheel={onWheel} onMouseDown={onDown} onMouseMove={onMove}
      onMouseUp={onUp} onMouseLeave={onUp} onDoubleClick={onDbl}>
      <canvas ref={canvasRef} style={{display:'block',width:'100%',height:'100%'}}/>
      {tip && (
        <div style={{ position:'absolute', left:tip.x+12, top:tip.y-28, pointerEvents:'none',
                      background:C.header, color:'#fff', borderRadius:4, padding:'3px 8px',
                      fontSize:10, fontFamily:'monospace', boxShadow:'0 2px 6px rgba(0,0,0,0.2)' }}>
          [{tip.gx},{tip.gy}] = {tip.v}
        </div>
      )}
      <div style={{ position:'absolute', bottom:60, left:8, fontSize:9, color: C.textMid,
                    fontFamily:'monospace', lineHeight:1.6, pointerEvents:'none' }}>
        <div>Scroll: zoom · Sürükle: pan · Çift tık: sıfırla</div>
        {sweeping && <div style={{color:C.accent}}>⏩ Animasyon aktif</div>}
      </div>
      <ColorBar vmin={colorRange?.min??0} vmax={colorRange?.max??1} label="Cevher [0-1]" />
    </div>
  );
}

// ─── REE Anomali haritası (2D canvas) ─────────────────────────────────────────
function AnomalyMap({ modelData, colorRange }) {
  const canvasRef=useRef();
  useEffect(() => {
    if (!modelData?.length) return;
    const n=modelData.length;
    const cv=canvasRef.current;
    if (!cv) return;
    cv.width=n; cv.height=n;
    const ctx=cv.getContext('2d');
    const img=ctx.createImageData(n,n);
    const vmin=colorRange?.min??0, vmax=colorRange?.max??1;
    // Z ekseni boyunca maksimum değer projeksiyon (max intensity projection)
    for (let row=0;row<n;row++) {
      for (let col=0;col<n;col++) {
        let maxV=0;
        for (let z=0;z<n;z++) maxV=Math.max(maxV,modelData[col]?.[n-1-row]?.[z]??0);
        const t=Math.max(0,Math.min(1,(maxV-vmin)/(vmax-vmin+1e-9)));
        const [r,g,b]=scalarToColor(t).map(c=>Math.round(c*255));
        const px=(row*n+col)*4;
        img.data[px]=r;img.data[px+1]=g;img.data[px+2]=b;img.data[px+3]=255;
      }
    }
    ctx.putImageData(img,0,0);
  },[modelData,colorRange]);

  return (
    <div style={{position:'relative',width:'100%',height:'100%',background:'#EFF6FF',
                 display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',padding:16}}>
      <div style={{fontSize:11,fontWeight:700,color:C.header,marginBottom:8,
                   textTransform:'uppercase',letterSpacing:'0.06em'}}>
        REE Anomali Haritası — Max Intensity Projeksiyon (Z)
      </div>
      <canvas ref={canvasRef} style={{imageRendering:'pixelated',maxWidth:'85%',maxHeight:'75%',
                                      aspectRatio:'1',border:`2px solid ${C.border}`,borderRadius:4,
                                      boxShadow:'0 2px 12px rgba(0,0,0,0.1)'}}/>
      <div style={{marginTop:8,fontSize:9,color:C.textMid,fontFamily:'monospace'}}>
        Enlem (Y) · Boylam (X) — 480m × 480m domain
      </div>
      <ColorBar vmin={colorRange?.min??0} vmax={colorRange?.max??1} label="Maks. Cevher" />
    </div>
  );
}

// ─── İstatistik paneli ────────────────────────────────────────────────────────
function StatsPanel({ modelData, colorRange, jiHistory, jiCorrelation, jiSummary }) {
  const stats = useMemo(() => {
    if (!modelData?.length) return null;
    const flat=modelData.flat(2);
    const n=flat.length;
    const mean=flat.reduce((a,b)=>a+b,0)/n;
    const std=Math.sqrt(flat.reduce((a,b)=>a+(b-mean)**2,0)/n);
    const above50=flat.filter(v=>v>0.5).length;
    const above30=flat.filter(v=>v>0.3).length;
    const sorted=[...flat].sort((a,b)=>a-b);
    const p10=sorted[Math.floor(n*0.1)], p50=sorted[Math.floor(n*0.5)], p90=sorted[Math.floor(n*0.9)];
    // Histogram
    const bins=20, binW=(colorRange.max-colorRange.min)/bins;
    const hist=Array(bins).fill(0);
    flat.forEach(v=>{const b=Math.min(bins-1,Math.floor((v-colorRange.min)/binW));if(b>=0)hist[b]++;});
    const histData=hist.map((c,i)=>({bin:`${(colorRange.min+i*binW).toFixed(2)}`, count:c}));
    return { mean, std, above50, above30, p10, p50, p90, histData, n };
  }, [modelData, colorRange]);

  if (!stats) return (
    <div style={{padding:24,textAlign:'center',color:C.textLow,fontSize:13}}>
      İstatistik için önce bir analiz çalıştırın.
    </div>
  );

  return (
    <div style={{padding:12,overflowY:'auto',height:'100%'}}>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8,marginBottom:12}}>
        {[
          {label:'Ortalama', value:stats.mean.toFixed(4), color:C.teal},
          {label:'Std Sapma', value:stats.std.toFixed(4), color:C.purple},
          {label:'P10', value:stats.p10?.toFixed(4), color:C.textMid},
          {label:'P50 (Medyan)', value:stats.p50?.toFixed(4), color:C.accent},
          {label:'P90', value:stats.p90?.toFixed(4), color:C.err},
          {label:'>0.5 Hücre', value:`${stats.above50} / ${stats.n}`, color:C.ok},
        ].map(({label,value,color})=>(
          <div key={label} style={{background:C.surface,border:`1px solid ${C.border}`,
                                   borderRadius:6,padding:'8px 10px'}}>
            <div style={{fontSize:9,color:C.textLow,textTransform:'uppercase',letterSpacing:'0.05em'}}>{label}</div>
            <div style={{fontSize:14,fontWeight:700,fontFamily:'monospace',color}}>{value}</div>
          </div>
        ))}
      </div>

      <div style={{marginBottom:12}}>
        <div style={{fontSize:11,fontWeight:700,color:C.header,marginBottom:6,
                     textTransform:'uppercase',letterSpacing:'0.05em'}}>Değer Dağılımı</div>
        <ResponsiveContainer width="100%" height={120}>
          <BarChart data={stats.histData} margin={{top:0,right:4,left:-20,bottom:0}}>
            <XAxis dataKey="bin" tick={{fontSize:8,fill:C.textLow}} interval={4}/>
            <YAxis tick={{fontSize:8,fill:C.textLow}}/>
            <Tooltip contentStyle={{fontSize:10,background:C.surface,border:`1px solid ${C.border}`}}/>
            <Bar dataKey="count" fill={C.teal} radius={[2,2,0,0]}/>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {jiHistory.length > 0 && (
        <div style={{marginBottom:12}}>
          <div style={{fontSize:11,fontWeight:700,color:C.header,marginBottom:6,
                       textTransform:'uppercase',letterSpacing:'0.05em'}}>Yakınsama</div>
          <ResponsiveContainer width="100%" height={100}>
            <AreaChart data={jiHistory} margin={{top:0,right:4,left:-20,bottom:0}}>
              <defs>
                <linearGradient id="mGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={C.teal} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={C.teal} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="iter" tick={{fontSize:8}} hide/>
              <YAxis tick={{fontSize:8,fill:C.textLow}}/>
              <Tooltip contentStyle={{fontSize:10}}/>
              <Area type="monotone" dataKey="misfit" stroke={C.teal} fill="url(#mGrad)" dot={false} strokeWidth={2}/>
            </AreaChart>
          </ResponsiveContainer>
          {jiSummary && (
            <div style={{display:'flex',gap:8,marginTop:6}}>
              <Tag color={C.ok}>Misfit {jiSummary.final?.toFixed(4)}</Tag>
              <Tag color={C.teal}>RMSE {jiSummary.rmse?.toFixed(4)}</Tag>
            </div>
          )}
        </div>
      )}

      {Object.keys(jiCorrelation).length > 0 && (
        <div>
          <div style={{fontSize:11,fontWeight:700,color:C.header,marginBottom:6,
                       textTransform:'uppercase',letterSpacing:'0.05em'}}>Korelasyon (Pearson)</div>
          {Object.entries(jiCorrelation).map(([k,v])=>(
            <div key={k} style={{display:'flex',justifyContent:'space-between',alignItems:'center',
                                  padding:'5px 0',borderBottom:`1px solid ${C.border}`}}>
              <span style={{fontSize:11,color:C.textMid}}>
                {k.replace('_',' ↔ ').replace('grav','Grav').replace('mag','Mag').replace('csamt','CSAMT')}
              </span>
              <div style={{display:'flex',alignItems:'center',gap:6}}>
                <div style={{width:50,height:6,background:C.border,borderRadius:3,overflow:'hidden'}}>
                  <div style={{width:`${Math.abs(v)*100}%`,height:'100%',borderRadius:3,
                                background:Math.abs(v)>0.6?C.ok:Math.abs(v)>0.3?C.accent:C.textLow}}/>
                </div>
                <span style={{fontSize:11,fontFamily:'monospace',fontWeight:700,
                              color:Math.abs(v)>0.6?C.ok:Math.abs(v)>0.3?C.accent:C.textMid}}>
                  {v.toFixed(3)}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Geometri Üretici paneli ──────────────────────────────────────────────────
const GEOM_TYPES = [
  { id:'beylikova_vein', label:'Hidrotermal Damar', icon:'⛏', color:'#B45309',
    short:'KB-GD eğimli damar + breş + halo',
    detail:'Beylikova (Eskişehir) REE-F-Ba-Th yatağı tipi. Karbonatlı kayaçlarda, yüksek yoğunluk+düşük özdirenç.' },
  { id:'pipe', label:'Borumsu / Pipe', icon:'🔵', color:'#1D4ED8',
    short:'Düşey silindirik yapı',
    detail:'Kimberlite (elmas), Cu-Mo porfiri, alkalik sistemler. Gravite+IP odaklı anomali.' },
  { id:'lens', label:'Mercek / Lens', icon:'🔶', color:'#065F46',
    short:'Yatay elipsoidal mercek',
    detail:'SEDEX (Zn-Pb-Ag), VMS (Cu-Zn), tabaka uyumlu Au-Ag. CSAMT derinlik kontrolü.' },
  { id:'stratabound', label:'Katmana Bağlı', icon:'📐', color:'#7C3AED',
    short:'Yatay katman, kenarda yoğunlaşma',
    detail:'Sedimantta Cu-Co, PGE reef, karbonatlarda Zn-Pb. Gravite+mag ayrımı zor.' },
];

function GeometryPanel({ onGenerated, log, apiBase }) {
  const [geomType, setGeomType]=useState('beylikova_vein');
  const [nbc, setNbc]=useState(32);
  const [dip, setDip]=useState(60);
  const [depthTop, setDepthTop]=useState(30);
  const [depthBot, setDepthBot]=useState(380);
  const [width, setWidth]=useState(60);
  const [breccia, setBreccia]=useState(true);
  const [halo, setHalo]=useState(true);
  const [loading, setLoading]=useState(false);
  const [result, setResult]=useState(null);

  const selected = GEOM_TYPES.find(g=>g.id===geomType);

  const generate = async () => {
    setLoading(true);
    try {
      // 1) Geometriyi backend'de üret
      const r = await apiFetch(`${apiBase}/api/generate-geometry`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({geom_type:geomType, nbc, dip_deg:dip,
          depth_top_m:depthTop, depth_bot_m:depthBot, width_m:width,
          add_breccia:breccia, add_halo:halo, seed:42}),
      });
      const d = await r.json();
      setResult(d);

      // 2) Backend'den save-geometry endpoint'iyle .npy olarak kaydet
      const saveR = await apiFetch(`${apiBase}/api/save-geometry`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          model_data: d.model_data,
          filename: `Y_${geomType}_${nbc}x${nbc}x${nbc}.npy`,
        }),
      });
      const saveD = await saveR.json();

      // 3) Üst bileşene bildir — dataset seçimini otomatik yap
      onGenerated(d, saveD.filename);
      log('ok', `Geometri üretildi ve kaydedildi: ${saveD.filename} | ${d.stats.active_voxels} aktif voksel`);
    } catch(e) { log('err', `Geometri hatası: ${e.message}`); }
    finally { setLoading(false); }
  };

  return (
    <div style={{fontSize:12}}>
      {/* Geometri tipi seçimi */}
      <div style={{marginBottom:12}}>
        <div style={{fontSize:11,fontWeight:700,color:C.header,textTransform:'uppercase',
                     letterSpacing:'0.06em',marginBottom:8}}>Prospeksiyon Geometrisi</div>
        <div style={{display:'flex',flexDirection:'column',gap:5}}>
          {GEOM_TYPES.map(g=>(
            <button key={g.id} onClick={()=>setGeomType(g.id)}
              style={{display:'flex',alignItems:'flex-start',gap:8,padding:'8px 10px',
                      borderRadius:6,cursor:'pointer',textAlign:'left',
                      border:`1.5px solid ${geomType===g.id?g.color:C.border}`,
                      background:geomType===g.id?`${g.color}10`:C.surface,
                      transition:'all 0.15s'}}>
              <span style={{fontSize:16,lineHeight:1,flexShrink:0}}>{g.icon}</span>
              <div>
                <div style={{fontSize:11,fontWeight:700,color:geomType===g.id?g.color:C.text}}>{g.label}</div>
                <div style={{fontSize:10,color:C.textLow,marginTop:1}}>{g.short}</div>
              </div>
            </button>
          ))}
        </div>
        {selected && (
          <div style={{marginTop:8,padding:'8px 10px',background:`${C.teal}08`,
                       border:`1px solid ${C.teal}30`,borderRadius:6,
                       fontSize:10,color:C.textMid,lineHeight:1.6}}>
            {selected.detail}
          </div>
        )}
      </div>

      {/* Parametreler */}
      <div style={{borderTop:`1px solid ${C.border}`,paddingTop:10,marginBottom:10}}>
        <div style={{fontSize:11,fontWeight:700,color:C.header,textTransform:'uppercase',
                     letterSpacing:'0.06em',marginBottom:8}}>Parametreler</div>

        <div style={{display:'flex',gap:6,marginBottom:8}}>
          <div style={{flex:1}}>
            <div style={{fontSize:10,color:C.textMid,marginBottom:3}}>Grid çözünürlüğü</div>
            <select value={nbc} onChange={e=>setNbc(parseInt(e.target.value))}
              style={{width:'100%',padding:'5px 8px',border:`1px solid ${C.border}`,
                      borderRadius:5,background:C.surface,color:C.text,fontSize:11}}>
              <option value={16}>16³ (30m/voksel)</option>
              <option value={32}>32³ (15m/voksel)</option>
              <option value={64}>64³ (7.5m/voksel)</option>
            </select>
          </div>
        </div>

        {geomType==='beylikova_vein' && (
          <RangeRow label="Eğim açısı (dip)" value={dip} min={20} max={90} step={5}
            onChange={setDip} format={v=>`${Math.round(v)}°`}/>
        )}
        <RangeRow label="Üst derinlik" value={depthTop} min={0} max={200} step={10}
          onChange={setDepthTop} format={v=>`${Math.round(v)} m`}/>
        <RangeRow label="Alt derinlik" value={depthBot} min={50} max={470} step={10}
          onChange={setDepthBot} format={v=>`${Math.round(v)} m`}/>
        <RangeRow label="Genişlik / yarıçap" value={width} min={10} max={150} step={5}
          onChange={setWidth} format={v=>`${Math.round(v)} m`}/>

        <div style={{display:'flex',gap:12,marginTop:6}}>
          {[{id:'breccia',label:'Breş zonu',val:breccia,set:setBreccia},
            {id:'halo',label:'Alterasyon halo',val:halo,set:setHalo}].map(({id,label,val,set})=>(
            <label key={id} style={{display:'flex',alignItems:'center',gap:5,cursor:'pointer',fontSize:11,color:C.textMid}}>
              <input type="checkbox" checked={val} onChange={e=>set(e.target.checked)} style={{accentColor:C.teal}}/>
              {label}
            </label>
          ))}
        </div>
      </div>

      <Btn onClick={generate} disabled={loading} variant="teal" style={{width:'100%',marginBottom:10}}
        icon={loading?Loader2:Mountain}>
        {loading?'Üretiliyor...':'Geometri Oluştur'}
      </Btn>

      {/* Petrofizik değerler */}
      {result && (
        <div style={{background:C.bg,border:`1px solid ${C.border}`,borderRadius:6,padding:10}}>
          <div style={{fontSize:11,fontWeight:700,color:C.header,marginBottom:8,
                       textTransform:'uppercase',letterSpacing:'0.06em'}}>Petrofizik Değerler</div>

          {/* Yoğunluk */}
          {[
            {label:'Yoğunluk', host:`${result.petrophys.density_contrast_gcm3.host} g/cm³`,
             ore:`${result.petrophys.density_contrast_gcm3.ore_max} g/cm³`,
             formula:result.petrophys.density_contrast_gcm3.formula,
             note:result.petrophys.density_contrast_gcm3.note, color:C.teal},
            {label:'Süseptibilite', host:`${result.petrophys.susceptibility_SI.host.toExponential(0)} SI`,
             ore:`${result.petrophys.susceptibility_SI.ore_max.toExponential(1)} SI`,
             formula:result.petrophys.susceptibility_SI.formula,
             note:result.petrophys.susceptibility_SI.note, color:C.accent},
            {label:'Özdirenç', host:`${result.petrophys.resistivity_ohmm.host} Ω·m`,
             ore:`${result.petrophys.resistivity_ohmm.ore_min} Ω·m`,
             formula:result.petrophys.resistivity_ohmm.formula,
             note:result.petrophys.resistivity_ohmm.note, color:C.purple},
          ].map(({label,host,ore,formula,note,color})=>(
            <div key={label} style={{marginBottom:8,padding:'7px 8px',background:C.surface,
                                     borderRadius:5,border:`1px solid ${color}30`}}>
              <div style={{fontSize:10,fontWeight:700,color,marginBottom:4}}>{label}</div>
              <div style={{display:'flex',gap:8,fontSize:10,fontFamily:'monospace'}}>
                <div><span style={{color:C.textLow}}>Host: </span><span>{host}</span></div>
                <div><span style={{color:C.textLow}}>Cevher: </span><span style={{color}}>{ore}</span></div>
              </div>
              <div style={{fontSize:9,color:C.textLow,marginTop:3,fontFamily:'monospace'}}>{formula}</div>
              <div style={{fontSize:9,color:C.textLow,marginTop:1,fontStyle:'italic'}}>{note}</div>
            </div>
          ))}

          {/* Model istatistikleri */}
          <div style={{display:'flex',gap:6,flexWrap:'wrap',marginTop:6}}>
            <Tag color={C.ok}>Cevher: {(result.stats.ore_fraction*100).toFixed(1)}%</Tag>
            <Tag color={C.teal}>Halo: {(result.stats.halo_fraction*100).toFixed(1)}%</Tag>
            <Tag color={C.textMid}>Grid: {result.params.grid}</Tag>
            <Tag color={C.textMid}>{result.params.voxel_m?.toFixed(1)}m/voksel</Tag>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Filtreleme & Gridleme paneli ──────────────────────────────────────────────
function FilterPanel({ modelData, onFiltered }) {
  const [thresh, setThresh]=useState(0.3);
  const [smooth, setSmooth]=useState(0);
  const [clampMin, setClampMin]=useState(0);
  const [clampMax, setClampMax]=useState(1);

  const applyFilter = () => {
    if (!modelData?.length) return;
    const filtered=modelData.map(plane=>
      plane.map(row=>
        row.map(v=>{
          let fv=Math.max(clampMin,Math.min(clampMax,v));
          if (fv<thresh) fv=0;
          return fv;
        })
      )
    );
    onFiltered(filtered);
  };

  return (
    <div style={{padding:4}}>
      <RangeRow label="Eşik değeri (threshold)" value={thresh} min={0} max={1} step={0.01}
        onChange={setThresh} format={v=>v.toFixed(2)}/>
      <RangeRow label="Kırp alt sınır" value={clampMin} min={0} max={1} step={0.01}
        onChange={setClampMin} format={v=>v.toFixed(2)}/>
      <RangeRow label="Kırp üst sınır" value={clampMax} min={0} max={1} step={0.01}
        onChange={setClampMax} format={v=>v.toFixed(2)}/>
      <div style={{marginTop:8}}>
        <Btn onClick={applyFilter} icon={Filter} size="sm" variant="teal" style={{width:'100%'}}>
          Filtreyi Uygula
        </Btn>
      </div>
      <div style={{marginTop:8,fontSize:10,color:C.textLow,lineHeight:1.5}}>
        Eşik altı vokseller sıfırlanır. Değişiklikler geçicidir — orijinal veri korunur.
      </div>
    </div>
  );
}

// ─── Dışa aktarma ─────────────────────────────────────────────────────────────
function ExportPanel({ modelData, colorRange, logs }) {
  const exportPNG = () => {
    if (!modelData?.length) return;
    const n=modelData.length;
    const cv=document.createElement('canvas'); cv.width=n; cv.height=n;
    const ctx=cv.getContext('2d');
    const img=ctx.createImageData(n,n);
    const vmin=colorRange.min, vmax=colorRange.max;
    for (let row=0;row<n;row++)
      for (let col=0;col<n;col++) {
        let maxV=0;
        for (let z=0;z<n;z++) maxV=Math.max(maxV,modelData[col]?.[n-1-row]?.[z]??0);
        const t=Math.max(0,Math.min(1,(maxV-vmin)/(vmax-vmin+1e-9)));
        const [r,g,b]=scalarToColor(t).map(c=>Math.round(c*255));
        const px=(row*n+col)*4;
        img.data[px]=r;img.data[px+1]=g;img.data[px+2]=b;img.data[px+3]=255;
      }
    ctx.putImageData(img,0,0);
    const a=document.createElement('a'); a.download='geopinn_anomaly.png';
    a.href=cv.toDataURL(); a.click();
  };

  const exportCSV = () => {
    if (!modelData?.length) return;
    const n=modelData.length;
    const rows=['x,y,z,value'];
    for (let x=0;x<n;x++)
      for (let y=0;y<n;y++)
        for (let z=0;z<n;z++) {
          const v=modelData[x]?.[y]?.[z];
          if (v>0.05) rows.push(`${x},${y},${z},${v.toFixed(6)}`);
        }
    const blob=new Blob([rows.join('\n')],{type:'text/csv'});
    const a=document.createElement('a'); a.download='geopinn_model.csv';
    a.href=URL.createObjectURL(blob); a.click();
  };

  const exportLog = () => {
    const blob=new Blob([logs.map(l=>`${l.t} [${l.level}] ${l.msg}`).join('\n')],{type:'text/plain'});
    const a=document.createElement('a'); a.download='geopinn_log.txt';
    a.href=URL.createObjectURL(blob); a.click();
  };

  return (
    <div style={{display:'flex',flexDirection:'column',gap:8,padding:4}}>
      <Btn onClick={exportPNG} icon={Download} variant="secondary" size="sm">
        Anomali Haritası (PNG)
      </Btn>
      <Btn onClick={exportCSV} icon={Download} variant="secondary" size="sm">
        3D Model Noktaları (CSV)
      </Btn>
      <Btn onClick={exportLog} icon={FileText} variant="secondary" size="sm">
        Log Kaydı (TXT)
      </Btn>
      <div style={{fontSize:10,color:C.textLow,marginTop:4,lineHeight:1.5}}>
        CSV sadece eşik {'>'} 0.05 vokselleri içerir. GeoTIFF için QGIS'e CSV import edin.
      </div>
    </div>
  );
}

// ─── Ana uygulama ──────────────────────────────────────────────────────────────
// ── API fetch wrapper — ngrok warning sayfasını atlar ────────────────────────
async function apiFetch(url, options = {}) {
  const headers = {
    'ngrok-skip-browser-warning': 'true',
    ...(options.headers || {}),
  };
  return fetch(url, { ...options, headers });
}
// Electron: window.electronAPI.getConfig() → {colabUrl, backendMode}
// Tarayıcı dev modu: localhost fallback
const _isElectron = typeof window !== 'undefined' && !!window.electronAPI;
let API_BASE = 'http://127.0.0.1:8000';   // varsayılan, useEffect'te güncellenir

export default function App() {
  const [theme, toggleTheme] = useTheme();
  // Veri
  const [modelData, setModelData]=useState(null);
  const [filteredData, setFilteredData]=useState(null);
  const [colorRange, setColorRange]=useState({min:0,max:1});
  const [metrics, setMetrics]=useState({mass:0,volume:0,heat:0});
  const [results, setResults]=useState({});

  // Görünüm
  const [viewMode, setViewMode]=useState('3d'); // '3d' | 'slice' | 'anomaly' | 'stats'
  const [engineMode, setEngineMode]=useState('prism'); // 'prism' | 'fvm'
  const [fvmResult, setFvmResult]=useState(null);
  const [fvmRunning, setFvmRunning]=useState(false);
  const [fvmAvailable, setFvmAvailable]=useState(null);
  // Radyometri & ısı akışı
  const [radResult, setRadResult]=useState(null);
  const [radRunning, setRadRunning]=useState(false);
  const [radAvailable, setRadAvailable]=useState(false);
  const [radParams, setRadParams]=useState({
    u_bg:3.0, u_ore:15.0, th_bg:12.0, th_ore:60.0,
    k_bg:2.5, k_ore:4.5, k_thermal:2.5,
  });
  // Veri formatı seçici
  const [dataFormats, setDataFormats]=useState({});
  const [selectedFormat, setSelectedFormat]=useState('auto');
  const [viewBg, setViewBg]=useState('light');
  const [isoThreshold, setIsoThreshold]=useState(0.08);
  const [opacity3d, setOpacity3d]=useState(1.0);
  const [showTopo, setShowTopo]=useState(false);
  const [topoOpacity, setTopoOpacity]=useState(0.6);
  const [sliceAxis, setSliceAxis]=useState('z');
  const [sliceIdx, setSliceIdx]=useState(8);
  const [sweeping, setSweeping]=useState(false);
  const sweepRef=useRef(null);

  // Paneller
  const [rightTab, setRightTab]=useState('stats');
  const [leftTab, setLeftTab]=useState('layers');

  // Backend bağlantı yönetimi
  const [apiBase, setApiBase]           = useState('http://127.0.0.1:8000');
  const [backendMode, setBackendMode]   = useState('local');  // 'local' | 'colab'
  const [colabUrl, setColabUrl]         = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [settingsSaved, setSettingsSaved] = useState(false);

  // Electron'dan runtime config oku
  useEffect(() => {
    if (window.electronAPI) {
      window.electronAPI.getConfig().then(cfg => {
        if (cfg.colabUrl && cfg.backendMode === 'colab') {
          const cleanUrl = cfg.colabUrl.replace(/\/+$/, '');
          setApiBase(cleanUrl);
          setBackendMode('colab');
          setColabUrl(cleanUrl);
        }
      }).catch(() => {});
    }
  }, []);

  const saveSettings = async () => {
    const cleanUrl = colabUrl.replace(/\/+$/, '');  // sondaki slash'ı sil
    const newBase = backendMode === 'colab' && cleanUrl ? cleanUrl : 'http://127.0.0.1:8000';
    setApiBase(newBase);
    setColabUrl(cleanUrl);
    if (window.electronAPI) {
      await window.electronAPI.saveConfig({ colabUrl, backendMode });
      if (backendMode === 'local') await window.electronAPI.restartBackend();
    }
    setSettingsSaved(true);
    setTimeout(() => setSettingsSaved(false), 2000);
    setShowSettings(false);
    log('ok', `Backend: ${backendMode === 'colab' ? colabUrl : 'Yerel (localhost:8000)'}`);
  };

  // Yardım modalı
  const [showHelp,   setShowHelp]   = useState(false);
  const [helpTopic,  setHelpTopic]  = useState('workflow');

  const HELP = {
    values: {
      title: '🎨 Model Değerleri [0–1]',
      items: [
        { label:'0.0',     color:'#3B4CB8',
          desc:'Saf host kaya — granit, kireçtaşı, şist\nGravite anomalisi yok, manyetik anomali yok' },
        { label:'0.1–0.3', color:'#2D7D52',
          desc:'Alterasyon halo / zayıf mineralleşme\nDüşük anomali, CSAMT\'ta hafif iletkenlik artışı' },
        { label:'0.5–0.8', color:'#D97706',
          desc:'Cevher zonu — yoğun mineralleşme\nBelirgin gravite + manyetik anomali' },
        { label:'1.0',     color:'#C53030',
          desc:'Saf cevher — REE-florit-barit damarı\nρ=4.70 g/cm³ · χ=4×10⁻⁴ SI · ρₑ=50 Ω·m' },
      ]
    },
    simpeg: {
      title: '⚙️ SimPEG Parametreleri',
      items: [
        { label:'α_s = 10⁻⁵', color:C.err,
          desc:'Çok küçük → Detaylı ama gürültülü model' },
        { label:'α_s = 10⁻⁴', color:C.ok,
          desc:'✓ Dengeli başlangıç noktası (önerilen)' },
        { label:'α_s = 10⁻²', color:C.textMid,
          desc:'Büyük → Çok düzgün, küçük, basit model' },
        { label:'α_x = 10⁰',  color:C.ok,
          desc:'✓ Dengeli yumuşaklık (önerilen)' },
        { label:'16³ grid',    color:C.teal,
          desc:'Hızlı test (30m/voksel) — ~1-2 dk' },
        { label:'32³ grid',    color:C.accent,
          desc:'Sonuç kalitesi (15m/voksel) — ~3-5 dk' },
      ]
    },
    geom: {
      title: '🗺️ Geometri Tipi Seçimi',
      items: [
        { label:'Beylikova Damar', color:'#B45309',
          desc:'REE-F-Ba-Th, Au-Ag, skarn, hidrotermal\ndip=50-70° · genişlik=40-80m · derinlik=30-450m' },
        { label:'Pipe / Borumsu', color:'#1D4ED8',
          desc:'Kimberlite, Cu-Mo porfiri, alkalik Cu-Au\nDüşey silindir, küçük alan, derin' },
        { label:'Lens / Mercek',  color:'#065F46',
          desc:'SEDEX (Zn-Pb-Ag), VMS (Cu-Zn), tabaka uyumlu\nYatay, geniş, sığ. CSAMT derinlik kontrolü kritik' },
        { label:'Stratabound',    color:'#7C3AED',
          desc:'Sedimantta Cu-Co, PGE, karbonatlarda Zn-Pb\nYatay katman, kenar yoğun. Gravite+mag ayrımı zor' },
      ]
    },
    radiometry: {
      title: '☢️ Radyometri & Isı Akışı',
      items: [
        { label:'Th/U > 4', color:'#D97706',
          desc:'Gelişmiş alterasyon — REE mobilizasyonu başlamış\nMonazit/xenotim Th ve U\'u konsantre eder\nBeylikova\'da tipik: Th/U = 4–8' },
        { label:'eU (efektif U)', color:'#B45309',
          desc:'eU = U_ppm + 0.335 × Th_ppm\n>8 ppm → yüksek radyojenik kaynak, güçlü anomali\nMonazit, uraninit, coffinit indikatörü' },
        { label:'Isı Akışı > 90 mW/m²', color:'#C53030',
          desc:'Aktif hidrotermal sistem veya radyojenik zengin granit\nBeylikova analogu: 80–120 mW/m² beklenir\nSP anomalisiyle örtüşmeli' },
        { label:'Bileşik Skor > 0.6', color:'#2D7D52',
          desc:'Th/U + eU + K anomalilerinin ağırlıklı ortalaması\n>0.6 → sondaj öncelikli hedef\n>0.8 → en yüksek öncelik' },
      ]
    },
    fvm: {
      title: '⚡ FVM vs Analitik Motor',
      items: [
        { label:'Analitik (Prizma)', color:C.teal,
          desc:'Nagy (gravite) / Bhattacharyya (manyetik) kapalı form\nSonsuz homojen uzay varsayımı · GPU hızlandırmalı\nHızlı, standart araştırma için yeterli' },
        { label:'FVM (Poisson)', color:C.accent,
          desc:'∇²U = kaynak denklemi, Dirichlet BC\nSınırlı domain, gerçekçi sınır koşulları\nYavaş ama domain sınırına yakın cevherlerde daha doğru' },
        { label:'Göreli RMSE < %5', color:C.ok,
          desc:'Motorlar uyumlu → analitik sonuç güvenilir\nFVM kullanmaya gerek yok' },
        { label:'Göreli RMSE > %5', color:C.err,
          desc:'Sınır etkisi var → FVM modunu kullan\nDomain kenarına yakın cevher gövdesi veya çok derin hedef' },
      ]
    },
    workflow: {
      title: '📋 Beylikova REE Arama Akışı',
      items: [
        { label:'① Geometri Oluştur',  color:C.teal,
          desc:'Sağ panel → Geometri → "Hidrotermal Damar" seç\ndip=60°, üst=30m, alt=380m, genişlik=60m → Oluştur' },
        { label:'② Forward Modelleme', color:C.accent,
          desc:'Sol panel → Gravite + Manyetik aktif → Analizi Başlat\nHarita sekmesinde anomali dağılımını gör' },
        { label:'③ Radyometri',        color:'#D97706',
          desc:'Sağ panel → Radyometri → parametreleri ayarla → Hesapla\nTh/U haritası + ısı akışı → REE hedef skoru' },
        { label:'④ Joint Inversion',   color:C.purple,
          desc:'Sol → Ters Çözüm → Grav+Mag+CSAMT → 32³ grid → 60 iter\nMisfit yakınsama grafiğini izle' },
        { label:'⑤ Belirsizlik',       color:C.ok,
          desc:'Sağ → Belirsizlik → 8 realizasyon → %5 gürültü\nCV<0.3 = güvenilir hedef bölgesi' },
        { label:'⑥ Dışa Aktar',       color:C.textMid,
          desc:'Sağ → Dışa Aktar → CSV (koordinatlar) + PNG (haritalar)\nSondaj lokasyon önerileri için kullan' },
      ]
    },
  };

  // Uncertainty
  const [uqRunning, setUqRunning]=useState(false);
  const [uqResult, setUqResult]=useState(null);
  const [uqDisplayMode, setUqDisplayMode]=useState('mean'); // mean|std|cv|p10|p90
  const [uqNReal, setUqNReal]=useState(5);
  const [uqNoise, setUqNoise]=useState(0.03);
  const [uqIter, setUqIter]=useState(40);

  // Katmanlar & dataset
  const [settings, setSettings]=useState({grav:true,mag:true,csamt:false,index:0});
  const [datasets, setDatasets]=useState([]);
  const [selY, setSelY]=useState(null);
  const [selGM, setSelGM]=useState(null);
  const [selCS, setSelCS]=useState(null);
  const [jiGridSize, setJiGridSize]=useState(16);
  const [uploading, setUploading]=useState(false);
  const fileRef=useRef();

  // Joint inversion
  const [jiOpen, setJiOpen]=useState(false);
  const [jiRunning, setJiRunning]=useState(false);
  const [jiWeights, setJiWeights]=useState({grav:1.0,mag:1.0,csamt:1.0});
  const [jiIter, setJiIter]=useState(60);
  const [jiHistory, setJiHistory]=useState([]);
  const [jiCorr, setJiCorr]=useState({});
  const [jiSummary, setJiSummary]=useState(null);

  // Analiz geçmişi
  const [lastRun, setLastRun]=useState(null);
  const [savedAnalyses, setSavedAnalyses]=useState([]);
  const [saving, setSaving]=useState(false);

  // Log
  const [logs, setLogs]=useState([{t:'00:00:00',level:'ok',msg:'GeoPINN Studio hazır.'}]);
  const [loading, setLoading]=useState(false);

  // Fullscreen
  const viewerRef=useRef();
  const [isFs, setIsFs]=useState(false);

  const ts=()=>new Date().toLocaleTimeString('tr-TR',{hour12:false});
  const log=(level,msg)=>setLogs(p=>[{t:ts(),level,msg},...p].slice(0,300));

  const activeDisplay = filteredData || modelData;

  // Sweep animasyonu
  const startSweep=()=>{
    if (sweepRef.current) return;
    setSweeping(true);
    let i=0, n=activeDisplay?.length??16;
    sweepRef.current=setInterval(()=>{
      setSliceIdx(i%n);
      i++;
    },120);
  };
  const stopSweep=()=>{
    clearInterval(sweepRef.current);
    sweepRef.current=null;
    setSweeping(false);
  };

  // Backend
  useEffect(()=>{
    apiFetch(`${apiBase}/api/health`).then(r=>r.json())
      .then(d=>log('ok',`Backend v${d.version} bağlandı.`))
      .catch(()=>log('err','Backend bağlantısı kurulamadı.'));
    fetchDatasets();
    fetchAnalyses();
  }, [apiBase]);

  const fetchDatasets=async()=>{
    try{const d=await(await apiFetch(`${apiBase}/api/data/list`)).json();setDatasets(d.files||[]);}
    catch(e){log('err','Veri listesi alınamadı.');}
  };
  const fetchAnalyses=async()=>{
    try{const d=await(await apiFetch(`${apiBase}/api/analyses`)).json();setSavedAnalyses(d.analyses||[]);}
    catch{}
  };

  const calcMetrics = (data) => {
    try {
      const vv = Math.pow(30, 3);
      let vol = 0, mass = 0, sum = 0;
      const flat = data.flat ? data.flat(2) : [];
      flat.forEach(v => {
        if (v > 0.1) { vol += vv; mass += v * 2000 * vv; sum += v; }
      });
      return { volume: vol, mass: mass / 1000, heat: sum * 0.05 };
    } catch(e) {
      return { volume: 0, mass: 0, heat: 0 };
    }
  };
  const updateModel = (data) => {
    if (!data || !data.length) return;
    // Derin iç içe array'i düzleştir ve yeniden şekillendir
    // Bu stack overflow'u önler
    try {
      setModelData(data);
      setFilteredData(null);
      // Flat array üzerinde çalış
      const flat = Array.isArray(data[0][0])
        ? data.flat(2)
        : data.flat ? data.flat(Infinity) : [];
      if (flat.length > 0) {
        setColorRange({ min: Math.min(...flat), max: Math.max(...flat) });
        setMetrics(calcMetrics(data));
      }
    } catch(e) {
      console.error('updateModel hatası:', e);
    }
  };

  const runAnalysis=async()=>{
    setLoading(true); log('info','Fizik motoru çağrılıyor...');
    try{
      const d=await(await apiFetch(`${apiBase}/api/run-physics-engine`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({grav_active:settings.grav,mag_active:settings.mag,
          csamt_active:settings.csamt,selected_index:settings.index,
          dataset:selY,dataset_gravmag:selGM,dataset_csamt:selCS,
          engine_mode:engineMode}),
      })).json();
      updateModel(d.model_data);
      setResults(d.results||{});
      setLastRun('physics');
      log('ok',`Analiz tamamlandı — ${d.dataset_used}`);
    } catch(e){log('err',`Analiz hatası: ${e.message}`);}
    finally{setLoading(false);}
  };

  // FVM + Radyometri status kontrolü
  useEffect(()=>{
    apiFetch(`${apiBase}/api/radiometry/status`).then(r=>r.json())
      .then(d=>setRadAvailable(d.available||false)).catch(()=>{});
    apiFetch(`${apiBase}/api/data/formats`).then(r=>r.json())
      .then(d=>setDataFormats(d.formats||{})).catch(()=>{});
  },[apiBase]);

  const runRadiometry=async()=>{
    setRadRunning(true); log('info','Radyometri & ısı akışı hesaplanıyor...');
    try{
      const d=await(await apiFetch(`${apiBase}/api/radiometry/forward`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          dataset:selY, selected_index:settings.index,
          u_background_ppm:radParams.u_bg,  u_ore_ppm:radParams.u_ore,
          th_background_ppm:radParams.th_bg, th_ore_ppm:radParams.th_ore,
          k_background_pct:radParams.k_bg,   k_ore_pct:radParams.k_ore,
          k_thermal:radParams.k_thermal,
          compute_heat_flow:true, compute_radiometry:true, compute_ree_index:true,
        }),
      })).json();
      setRadResult(d);
      setRightTab('radio');
      log('ok',`Radyometri tamamlandı — REE olasılık maks: ${d.ree_index?.stats?.max_prob?.toFixed(3)||'—'}`);
    }catch(e){log('err',`Radyometri hatası: ${e.message}`);}
    finally{setRadRunning(false);}
  };

  // FVM status kontrolü
  useEffect(()=>{
    apiFetch(`${apiBase}/api/fvm/status`).then(r=>r.json())
      .then(d=>setFvmAvailable(d.available||false)).catch(()=>setFvmAvailable(null));
  },[apiBase]);

  const runFvmCompare=async()=>{
    setFvmRunning(true); log('info','FVM vs Prizma karşılaştırması başlatıldı...');
    try{
      const d=await(await apiFetch(`${apiBase}/api/fvm/compare`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({dataset:selY,selected_index:settings.index,
          grav_active:settings.grav,mag_active:settings.mag}),
      })).json();
      setFvmResult(d);
      setRightTab('fvm');
      log('ok',`FVM karşılaştırma tamamlandı — Gravite RMSE: ${d.result?.gravity?.rmse_mgal?.toFixed(4)||'—'} mGal`);
    }catch(e){log('err',`FVM hatası: ${e.message}`);}
    finally{setFvmRunning(false);}
  };

  const runJI=async()=>{
    if(![settings.grav,settings.mag,settings.csamt].some(Boolean)){log('err','En az bir katman aktif olmalı.');return;}
    setJiRunning(true); log('info',`Joint inversion (${jiIter} iter, grid ${jiGridSize}³)...`);
    try{
      const d=await(await apiFetch(`${apiBase}/api/joint-inversion`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({grav_active:settings.grav,mag_active:settings.mag,
          csamt_active:settings.csamt,selected_index:settings.index,
          dataset:selY,dataset_grav_mag:selGM,dataset_csamt:selCS,
          n_iter:jiIter,weights:jiWeights,nbc_forward:jiGridSize}),
      })).json();
      updateModel(d.model_data);
      setJiHistory(d.history||[]);
      setJiCorr(d.correlation||{});
      setJiSummary({initial:d.initial_misfit,final:d.final_misfit,rmse:d.rmse_vs_true_model,dataset:d.dataset_used});
      setLastRun('joint');
      log('ok',`Joint inv. tamamlandı — misfit ${d.initial_misfit?.toFixed(3)}→${d.final_misfit?.toFixed(3)}`);
    }catch(e){log('err',`JI hatası: ${e.message}`);}
    finally{setJiRunning(false);}
  };

  // SimPEG
  const [simpegRunning, setSimpegRunning]=useState(false);
  const [simpegResult, setSimpegResult]=useState(null);
  const [simpegNbc, setSimpegNbc]=useState(16);
  const [simpegIter, setSimpegIter]=useState(15);
  const [simpegAlphaS, setSimpegAlphaS]=useState(-4);  // log10
  const [simpegAlphaX, setSimpegAlphaX]=useState(0);   // log10
  const [simpegAvailable, setSimpegAvailable]=useState(null);

  useEffect(()=>{
    apiFetch(`${apiBase}/api/simpeg/status`).then(r=>r.json())
      .then(d=>setSimpegAvailable(d.available))
      .catch(()=>setSimpegAvailable(null));  // null = bilinmiyor (false = kesin yok)
  },[apiBase]);

  const runSimPEG = async () => {
    setSimpegRunning(true);
    log('info',`SimPEG Tikhonov inversion başlatıldı (${simpegNbc}³, ${simpegIter} iter)...`);
    try {
      const r = await apiFetch(`${apiBase}/api/simpeg/inversion`,{
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          grav_active: settings.grav, mag_active: settings.mag,
          dataset: selY, dataset_grav_mag: selGM,
          selected_index: settings.index,
          nbc: simpegNbc, max_iter: simpegIter,
          alpha_s: Math.pow(10, simpegAlphaS),
          alpha_x: Math.pow(10, simpegAlphaX),
          chifact: 1.0,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setSimpegResult(d);
      if (d.model_data) { updateModel(d.model_data); setLastRun('joint'); }
      setRightTab('simpeg');
      log('ok',`SimPEG tamamlandı — ${d.method}`);
    } catch(e){ log('err',`SimPEG hatası: ${e.message}`); }
    finally{ setSimpegRunning(false); }
  };


  const runUQ = async () => {
    setUqRunning(true);
    log('info', `Belirsizlik analizi (${uqNReal} realizasyon, iter=${uqIter})...`);
    try {
      const r = await apiFetch(`${apiBase}/api/uncertainty`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          grav_active: settings.grav, mag_active: settings.mag, csamt_active: settings.csamt,
          dataset: selY, dataset_grav_mag: selGM, dataset_csamt: selCS,
          selected_index: settings.index,
          nbc_forward: jiGridSize, n_iter: uqIter,
          n_realizations: uqNReal, noise_level: uqNoise,
          weights: jiWeights, reg_lambda: 0.001,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setUqResult(d);
      // Ortalama modeli görüntüle
      updateModel(d.mean_model);
      setUqDisplayMode('mean');
      setRightTab('uq');
      log('ok', `Belirsizlik analizi tamamlandı — yüksek güven: %${d.summary.high_conf_pct}, düşük güven: %${d.summary.low_conf_pct}`);
    } catch(e) { log('err', `UQ hatası: ${e.message}`); }
    finally { setUqRunning(false); }
  };

  const applyUQLayer = (mode) => {
    if (!uqResult) return;
    const layers = { mean: uqResult.mean_model, std: uqResult.std_model,
                     cv: uqResult.cv_model, p10: uqResult.p10_model, p90: uqResult.p90_model };
    const data = layers[mode];
    if (!data) return;
    setUqDisplayMode(mode);
    updateModel(data);
    log('info', `Belirsizlik katmanı: ${mode.toUpperCase()}`);
  };

  const saveAnalysis = async () => {
    if(!lastRun){log('err','Önce bir analiz çalıştırın.');return;}
    const name=window.prompt('Analiz adı:',`${lastRun==='joint'?'Joint Inv.':'Analiz'} — ${new Date().toLocaleString('tr-TR')}`);
    if(!name) return;
    setSaving(true);
    try{
      await apiFetch(`${apiBase}/api/analyses`,{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({name,type:lastRun,dataset_used:selY,settings,results,metrics,
          model_data:modelData,history:jiHistory,correlation:jiCorr,summary:jiSummary})});
      log('ok',`Kaydedildi: "${name}"`); fetchAnalyses();
    }catch(e){log('err',`Kayıt hatası: ${e.message}`);}
    finally{setSaving(false);}
  };

  const loadAnalysis=async(id)=>{
    try{
      const r=await(await apiFetch(`${apiBase}/api/analyses/${id}`)).json();
      if(r.model_data){updateModel(r.model_data);}
      if(r.type==='joint'){setJiHistory(r.history||[]);setJiCorr(r.correlation||{});setJiSummary(r.summary||null);}
      if(r.settings)setSettings(r.settings);
      setLastRun(r.type); setRightTab('stats');
      log('ok',`Yüklendi: "${r.name}"`);
    }catch(e){log('err',`Yükleme hatası: ${e.message}`);}
  };

  const uploadFile=async(file)=>{
    if(!file?.name.endsWith('.npy')){log('err','Sadece .npy');return;}
    setUploading(true); log('info',`Yükleniyor: ${file.name}`);
    try{
      const form=new FormData(); form.append('file',file);
      const d=await(await fetch(`${apiBase}/api/data/upload`,{method:'POST',body:form,
        headers:{'ngrok-skip-browser-warning':'true'}})).json();
      log('ok',`Yüklendi: ${d.filename} (${(d.shape||[]).join('×')})`);
      await fetchDatasets();
      const fn=d.filename;
      if(fn.toLowerCase().startsWith('x_csamt')) setSelCS(fn);
      else if(fn.startsWith('X_')||fn.startsWith('x_')) setSelGM(fn);
      else setSelY(fn);
    }catch(e){log('err',`Yükleme: ${e.message}`);}
    finally{setUploading(false);}
  };

  const deleteDataset=async(fn)=>{
    await apiFetch(`${apiBase}/api/data/${encodeURIComponent(fn)}`,{method:'DELETE'});
    if(selY===fn)setSelY(null); if(selGM===fn)setSelGM(null); if(selCS===fn)setSelCS(null);
    fetchDatasets(); log('ok',`Silindi: ${fn}`);
  };

  useEffect(()=>{
    const h=()=>{setIsFs(!!document.fullscreenElement);setTimeout(()=>window.dispatchEvent(new Event('resize')),60);};
    document.addEventListener('fullscreenchange',h);
    return()=>document.removeEventListener('fullscreenchange',h);
  },[]);

  const toggleFs=()=>!document.fullscreenElement?viewerRef.current?.requestFullscreen():document.exitFullscreen();

  const layerConfig=[
    {key:'grav',label:'Gravite',sub:'Bouguer anomali',icon:Gauge,color:C.teal},
    {key:'mag',label:'Manyetik',sub:'TMI anomali',icon:Magnet,color:C.accent},
    {key:'csamt',label:'CSAMT',sub:'Görünür özdirenç',icon:Radio,color:C.purple},
  ];

  const viewTabs=[
    {id:'3d',label:'3D',icon:Box},
    {id:'slice',label:'Kesit',icon:Layers},
    {id:'anomaly',label:'Harita',icon:Map},
    {id:'stats',label:'İstatistik',icon:BarChart2},
  ];

  const rightTabs=[
    {id:'stats',  label:'Analiz',     icon:Activity},
    {id:'uq',     label:'Belirsizlik',icon:AlertCircle},
    {id:'simpeg', label:'SimPEG',     icon:GitCompare},
    {id:'fvm',    label:'FVM',        icon:Layers},
    {id:'radio',  label:'Radyometri', icon:Waves},
    {id:'geom',   label:'Geometri',   icon:Mountain},
    {id:'filter', label:'Filtre',     icon:Filter},
    {id:'export', label:'Dışa Aktar', icon:Download},
  ];

  // ─── Render ──────────────────────────────────────────────────────────────────
  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100vh',
                  background: C.bg, color: C.text,
                  fontFamily:'"Inter","Segoe UI",system-ui,sans-serif',
                  overflow:'hidden', fontSize:13,
                  transition:'background 0.2s, color 0.2s' }}>

      {/* ── Üst şerit ── */}
      <header style={{ height:42, background: theme==='dark' ? '#0D1018' : '#0D1117',
                       borderBottom:`1.5px solid ${C.accent}`,
                       display:'flex', alignItems:'center', padding:'0 14px', gap:12,
                       flexShrink:0, userSelect:'none' }}>
        {/* Logo */}
        <div style={{ display:'flex', alignItems:'center', gap:0, flexShrink:0 }}>
          {/* İmza element: sismik profil || çizgileri */}
          <div style={{ display:'flex', alignItems:'center', gap:2, marginRight:10 }}>
            {[6,10,14,10,6].map((h,i)=>(
              <div key={i} style={{ width:2, height:h, background:C.accent,
                borderRadius:1, opacity: i===2?1:0.5+i*0.05 }}/>
            ))}
          </div>
          <div>
            <span style={{ fontSize:13, fontWeight:800, color:'#fff',
              letterSpacing:'0.06em', fontFamily:'"Inter","Segoe UI",sans-serif' }}>GEO</span>
            <span style={{ fontSize:13, fontWeight:800, color:C.accent,
              letterSpacing:'0.06em' }}>PINN</span>
            <span style={{ fontSize:8, color:'rgba(255,255,255,0.35)',
              letterSpacing:'0.15em', textTransform:'uppercase',
              marginLeft:8, fontWeight:400 }}>STUDIO 3.0</span>
          </div>
        </div>

        {/* Dikey ayraç */}
        <div style={{ width:1, height:24, background:'rgba(255,255,255,0.12)', flexShrink:0 }}/>

        {/* Görünüm sekmeleri */}
        <div style={{ display:'flex', gap:1 }}>
          {viewTabs.map(({id,label,icon:Icon})=>(
            <button key={id} onClick={()=>setViewMode(id)}
              style={{ display:'flex', alignItems:'center', gap:4, padding:'4px 11px',
                       border:'none', cursor:'pointer', fontSize:11, fontWeight:600,
                       background: viewMode===id ? C.accent : 'transparent',
                       color: viewMode===id ? '#fff' : 'rgba(255,255,255,0.45)',
                       borderBottom: viewMode===id ? `2px solid ${C.accentL}` : '2px solid transparent',
                       transition:'all 0.12s', letterSpacing:'0.02em' }}>
              <Icon size={11}/>{label}
            </button>
          ))}
        </div>

        <div style={{ flex:1 }}/>

        {/* Durum */}
        <div style={{ display:'flex', alignItems:'center', gap:12, fontSize:11,
                      color:'rgba(255,255,255,0.7)', fontFamily:'monospace' }}>
          <span style={{ display:'flex', alignItems:'center', gap:5 }}>
            <span style={{ width:7, height:7, borderRadius:'50%',
                           background: loading||jiRunning ? C.warn : C.ok,
                           boxShadow: loading||jiRunning ? `0 0 6px ${C.warn}` : `0 0 6px ${C.ok}`,
                           display:'inline-block', animation: loading||jiRunning ? 'pulse 1s infinite' : 'none' }}/>
            {loading?'İşleniyor':jiRunning?'Ters çözüm':selGM||selY||'demo/sentetik'}
          </span>
          <span style={{ color:'rgba(255,255,255,0.4)' }}>|</span>
          <span>{layerConfig.filter(l=>settings[l.key]).length} katman</span>
          <span style={{ color:'rgba(255,255,255,0.4)' }}>|</span>
          <span style={{ fontSize:10, color: backendMode==='colab' ? '#34D399' : 'rgba(255,255,255,0.5)',
                         display:'flex', alignItems:'center', gap:4 }}>
            <span style={{ width:6, height:6, borderRadius:'50%', display:'inline-block',
                           background: backendMode==='colab' ? '#34D399' : 'rgba(255,255,255,0.3)' }}/>
            {backendMode==='colab' ? 'Colab GPU' : 'Yerel'}
          </span>
          {/* Tema toggle */}
          <button onClick={()=>toggleTheme()}
            title={theme==='dark'?'Aydınlık moda geç':'Karanlık moda geç'}
            style={{ background:'rgba(255,255,255,0.08)', border:'1px solid rgba(255,255,255,0.15)',
                     borderRadius:3, width:28, height:26, cursor:'pointer',
                     color:'rgba(255,255,255,0.7)', fontSize:13,
                     display:'flex', alignItems:'center', justifyContent:'center' }}>
            {theme==='dark' ? '☀' : '◑'}
          </button>
          <button onClick={()=>setShowSettings(true)}
            style={{ background:'rgba(255,255,255,0.08)', border:'1px solid rgba(255,255,255,0.15)',
                     borderRadius:3, padding:'3px 9px', cursor:'pointer', color:'rgba(255,255,255,0.7)',
                     fontSize:10, fontWeight:600, letterSpacing:'0.04em',
                     display:'flex', alignItems:'center', gap:4 }}>
            <Settings size={11}/> BAĞLANTI
          </button>
          <button onClick={()=>setShowHelp(true)}
            style={{ background:C.accent, border:'none', borderRadius:3,
                     padding:'3px 9px', cursor:'pointer', color:'#fff', fontSize:10,
                     fontWeight:700, letterSpacing:'0.04em',
                     display:'flex', alignItems:'center', gap:4 }}>
            <Info size={11}/> REHBER
          </button>
        </div>
      </header>

      {/* ── Yardım Modalı ── */}
      {showSettings && (
        <div style={{ position:'fixed', inset:0, zIndex:1000, background:'rgba(0,0,0,0.5)',
                      display:'flex', alignItems:'center', justifyContent:'center' }}
          onClick={()=>setShowSettings(false)}>
          <div style={{ background:C.surface, borderRadius:10, width:480,
                        boxShadow:'0 20px 60px rgba(0,0,0,0.3)', border:`1px solid ${C.border}` }}
            onClick={e=>e.stopPropagation()}>
            <div style={{ padding:'14px 18px', borderBottom:`1px solid ${C.border}`,
                          background:C.header, borderRadius:'10px 10px 0 0',
                          display:'flex', justifyContent:'space-between', alignItems:'center' }}>
              <span style={{ fontSize:14, fontWeight:700, color:'#fff' }}>Backend Bağlantısı</span>
              <button onClick={()=>setShowSettings(false)}
                style={{ background:'rgba(255,255,255,0.15)', border:'none', borderRadius:5,
                         padding:'2px 8px', cursor:'pointer', color:'#fff' }}>✕</button>
            </div>

            <div style={{ padding:20 }}>
              {/* Mod seçimi */}
              <div style={{ display:'flex', gap:8, marginBottom:16 }}>
                {[{id:'local',label:'🖥 Yerel Backend',desc:'Bilgisayarında çalışır, GPU yok'},
                  {id:'colab',label:'☁ Google Colab GPU',desc:'Ngrok URL gerekli, hızlı'}].map(({id,label,desc})=>(
                  <button key={id} onClick={()=>setBackendMode(id)}
                    style={{ flex:1, padding:'10px 12px', borderRadius:7, cursor:'pointer',
                             border:`2px solid ${backendMode===id?C.teal:C.border}`,
                             background: backendMode===id?`${C.teal}10`:C.bg,
                             textAlign:'left', transition:'all 0.15s' }}>
                    <div style={{ fontSize:12, fontWeight:700, color:backendMode===id?C.teal:C.text }}>{label}</div>
                    <div style={{ fontSize:10, color:C.textLow, marginTop:2 }}>{desc}</div>
                  </button>
                ))}
              </div>

              {/* Colab URL girişi */}
              {backendMode==='colab' && (
                <div style={{ marginBottom:16 }}>
                  <div style={{ fontSize:11, fontWeight:600, color:C.header, marginBottom:6 }}>
                    Ngrok URL (Colab'dan kopyala)
                  </div>
                  <input type="text" value={colabUrl}
                    onChange={e=>setColabUrl(e.target.value.trim())}
                    placeholder="https://xxxx-xx-xx.ngrok-free.app"
                    style={{ width:'100%', padding:'8px 10px', border:`1px solid ${C.border}`,
                             borderRadius:6, fontSize:12, fontFamily:'monospace',
                             background:C.bg, color:C.text, boxSizing:'border-box' }}/>
                  <div style={{ fontSize:10, color:C.textLow, marginTop:4, lineHeight:1.5 }}>
                    Colab notebook'ta "Backend URL:" satırının yanındaki URL'yi yapıştır.<br/>
                    Her yeni Colab oturumunda bu URL değişir.
                  </div>
                </div>
              )}

              {/* Bilgi kutusu */}
              <div style={{ background:`${C.teal}08`, border:`1px solid ${C.teal}25`,
                            borderRadius:6, padding:'8px 12px', marginBottom:16,
                            fontSize:10, color:C.textMid, lineHeight:1.6 }}>
                {backendMode==='local'
                  ? 'Yerel mod: Backend exe veya Python uvicorn ile localhost:8000\'de çalışır. GPU yok.'
                  : 'Colab modu: Tüm hesaplamalar GPU\'lu Colab sunucusunda yapılır. Colab notebook açık kalmalı.'}
              </div>

              {/* Kaydet butonu */}
              <div style={{ display:'flex', gap:8, justifyContent:'flex-end' }}>
                <Btn onClick={()=>setShowSettings(false)} variant="secondary" size="sm">İptal</Btn>
                <Btn onClick={saveSettings} variant="teal" size="sm" icon={settingsSaved?CheckCircle:Settings}>
                  {settingsSaved ? 'Kaydedildi!' : 'Kaydet ve Bağlan'}
                </Btn>
              </div>
            </div>
          </div>
        </div>
      )}
      {showHelp && (
        <div style={{ position:'fixed', inset:0, zIndex:1000, background:'rgba(0,0,0,0.6)',
                      display:'flex', alignItems:'center', justifyContent:'center' }}
          onClick={()=>setShowHelp(false)}>
          <div style={{ background:C.surface, borderRadius:12, width:600, maxHeight:'80vh',
                        overflow:'hidden', boxShadow:'0 24px 64px rgba(0,0,0,0.4)',
                        border:`1px solid ${C.border}` }}
            onClick={e=>e.stopPropagation()}>

            {/* Modal başlık */}
            <div style={{ padding:'14px 18px', borderBottom:`1px solid ${C.border}`,
                          display:'flex', alignItems:'center', justifyContent:'space-between',
                          background: C.header }}>
              <span style={{ fontSize:14, fontWeight:700, color:'#fff' }}>GeoPINN Kullanım Rehberi</span>
              <button onClick={()=>setShowHelp(false)}
                style={{ background:'rgba(255,255,255,0.15)', border:'none', borderRadius:5,
                         padding:'3px 8px', cursor:'pointer', color:'#fff', fontSize:12 }}>✕</button>
            </div>

            {/* Konu sekmeleri */}
            <div style={{ display:'flex', borderBottom:`1px solid ${C.border}`, background:C.bg }}>
              {[{id:'workflow',label:'İş Akışı'},{id:'values',label:'Değerler'},
                {id:'geom',label:'Geometri'},{id:'simpeg',label:'SimPEG'},
                {id:'radiometry',label:'Radyometri'},{id:'fvm',label:'FVM'}].map(({id,label})=>(
                <button key={id} onClick={()=>setHelpTopic(id)}
                  style={{ flex:1, padding:'9px 4px', border:'none', cursor:'pointer',
                           fontSize:11, fontWeight:700,
                           background: helpTopic===id ? C.surface : 'transparent',
                           color: helpTopic===id ? C.accent : C.textMid,
                           borderBottom: helpTopic===id ? `2px solid ${C.accent}` : '2px solid transparent' }}>
                  {label}
                </button>
              ))}
            </div>

            {/* İçerik */}
            <div style={{ padding:20, overflowY:'auto', maxHeight:'calc(80vh - 110px)' }}>
              <div style={{ fontSize:13, fontWeight:700, color:C.header, marginBottom:14 }}>
                {HELP[helpTopic]?.title}
              </div>
              <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
                {HELP[helpTopic]?.items.map(({label,color,desc})=>(
                  <div key={label} style={{ display:'flex', gap:12, padding:'10px 12px',
                                            background:`${color}08`, border:`1px solid ${color}25`,
                                            borderRadius:7, borderLeft:`3px solid ${color}` }}>
                    <div style={{ minWidth:120, flexShrink:0 }}>
                      <div style={{ fontSize:12, fontWeight:700, color, lineHeight:1.3 }}>{label}</div>
                    </div>
                    <div style={{ fontSize:11, color:C.textMid, lineHeight:1.7, whiteSpace:'pre-line' }}>
                      {desc}
                    </div>
                  </div>
                ))}
              </div>

              {helpTopic === 'workflow' && (
                <div style={{ marginTop:14, padding:'10px 12px', background:`${C.teal}08`,
                              border:`1px solid ${C.teal}25`, borderRadius:7,
                              fontSize:11, color:C.textMid, lineHeight:1.7 }}>
                  <strong style={{color:C.teal}}>Petrofizik formüller:</strong><br/>
                  {'ρ(x) = 2.70 + 2.00×f  [g/cm³]  →  f=1 → 4.70 g/cm³ (baritli REE cevheri)'}<br/>
                  {'χ(x) = 1×10⁻⁴ + 3×10⁻⁴×f  [SI]  →  f=1 → 4×10⁻⁴ SI'}<br/>
                  {'ρₑ(x) = 500 × 0.10^f  [Ω·m]  →  f=1 → 50 Ω·m (iletken sülfürlü zon)'}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Ana içerik ── */}
      <div style={{ display:'flex', flex:1, minHeight:0 }}>

        {/* ── Sol panel ── */}
        <aside style={{ width:260, background: C.surface, borderRight:`1px solid ${C.border}`,
                        display:'flex', flexDirection:'column', overflowY:'auto', flexShrink:0 }}>

          {/* Sol sekme */}
          <div style={{ display:'flex', borderBottom:`1px solid ${C.border}` }}>
            {[{id:'layers',label:'Katmanlar'},{id:'data',label:'Veri'},{id:'ji',label:'Ters Çözüm'}].map(({id,label})=>(
              <button key={id} onClick={()=>setLeftTab(id)}
                style={{ flex:1, padding:'8px 4px', border:'none', cursor:'pointer',
                         fontSize:10, fontWeight:700, letterSpacing:'0.05em', textTransform:'uppercase',
                         background: leftTab===id ? C.accent : 'transparent',
                         color: leftTab===id ? '#fff' : C.textMid,
                         borderBottom: leftTab===id ? `2px solid ${C.accent}` : '2px solid transparent',
                         transition:'all 0.15s' }}>
                {label}
              </button>
            ))}
          </div>

          {/* Katmanlar sekmesi */}
          {leftTab==='layers' && (
            <>
              <PanelSection title="Veri Katmanları" icon={Layers}>
                <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
                  {layerConfig.map(({key,label,sub,icon:Icon,color})=>(
                    <label key={key} style={{ display:'flex', alignItems:'center', gap:8, padding:'8px 10px',
                                              borderRadius:6, cursor:'pointer', transition:'all 0.15s',
                                              background: settings[key] ? `${color}12` : C.bg,
                                              border:`1px solid ${settings[key]?color+'40':C.border}` }}>
                      <input type="checkbox" checked={settings[key]}
                        onChange={e=>setSettings({...settings,[key]:e.target.checked})}
                        style={{accentColor:color,width:14,height:14}}/>
                      <Icon size={14} color={settings[key]?color:C.textLow}/>
                      <div>
                        <div style={{fontSize:12,fontWeight:600,color:settings[key]?C.text:C.textMid}}>{label}</div>
                        <div style={{fontSize:10,color:C.textLow}}>{sub}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </PanelSection>

              <PanelSection title="3D Görünüm" icon={Box}>
                <RangeRow label="İzoyüzey eşiği" value={isoThreshold} min={0.01} max={0.5} step={0.01}
                  onChange={setIsoThreshold} format={v=>`${(v*100).toFixed(0)}%`}/>
                <RangeRow label="Şeffaflık" value={opacity3d} min={0.1} max={1.0} step={0.05}
                  onChange={setOpacity3d} format={v=>`${(v*100).toFixed(0)}%`}/>
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
                  <span style={{fontSize:11,color:C.textMid}}>Arka plan</span>
                  <div style={{display:'flex',gap:4}}>
                    {['dark','light'].map(m=>(
                      <button key={m} onClick={()=>setViewBg(m)}
                        style={{padding:'3px 8px',borderRadius:4,border:`1px solid ${C.border}`,fontSize:10,
                                cursor:'pointer',background:viewBg===m?C.header:C.surface,
                                color:viewBg===m?'#fff':C.textMid}}>
                        {m==='dark'?'Koyu':'Açık'}
                      </button>
                    ))}
                  </div>
                </div>
                <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:8}}>
                  <input type="checkbox" id="topoCheck" checked={showTopo} onChange={e=>setShowTopo(e.target.checked)}
                    style={{accentColor:C.ok}}/>
                  <label htmlFor="topoCheck" style={{fontSize:11,color:C.textMid,cursor:'pointer'}}>Topografik katman</label>
                </div>
                {showTopo && (
                  <RangeRow label="Topografya saydamlığı" value={topoOpacity} min={0.1} max={1} step={0.05}
                    onChange={setTopoOpacity} format={v=>`${(v*100).toFixed(0)}%`}/>
                )}
              </PanelSection>

              <PanelSection title="Kesit Kontrol" icon={Grid} defaultOpen={false}>
                <div style={{display:'flex',gap:4,marginBottom:8}}>
                  {['x','y','z'].map(ax=>(
                    <button key={ax} onClick={()=>setSliceAxis(ax)}
                      style={{flex:1,padding:'4px',borderRadius:4,border:`1px solid ${sliceAxis===ax?C.teal:C.border}`,
                              background:sliceAxis===ax?`${C.teal}18`:'transparent',
                              color:sliceAxis===ax?C.teal:C.textMid,fontWeight:700,fontSize:11,cursor:'pointer'}}>
                      {ax.toUpperCase()}
                    </button>
                  ))}
                </div>
                <RangeRow label={`Kesit (${sliceAxis.toUpperCase()})`}
                  value={sliceIdx} min={0} max={(activeDisplay?.length??16)-1}
                  onChange={v=>{setSliceIdx(Math.round(v));setViewMode('slice');}}/>
                <div style={{display:'flex',gap:6}}>
                  {!sweeping
                    ? <Btn onClick={()=>{setViewMode('slice');startSweep();}} icon={Play} size="sm" variant="teal" style={{flex:1}}>Animasyon</Btn>
                    : <Btn onClick={stopSweep} icon={Sliders} size="sm" variant="danger" style={{flex:1}}>Durdur</Btn>
                  }
                </div>
              </PanelSection>
            </>
          )}

          {/* Veri sekmesi */}
          {leftTab==='data' && (
            <>
              <div style={{padding:12,borderBottom:`1px solid ${C.border}`}}>
                {/* Veri formatı seçici */}
                {Object.keys(dataFormats).length>0&&(
                  <div style={{marginBottom:8}}>
                    <div style={{fontSize:10,color:C.textMid,marginBottom:4,textTransform:'uppercase',letterSpacing:'0.05em'}}>
                      Veri formatı
                    </div>
                    <select value={selectedFormat}
                      onChange={e=>setSelectedFormat(e.target.value)}
                      style={{width:'100%',padding:'4px 6px',borderRadius:4,fontSize:11,
                        border:`1px solid ${C.border}`,background:C.bg,color:C.text,marginBottom:6}}>
                      <option value="auto">Otomatik tespit</option>
                      {Object.entries(dataFormats).map(([key,fmt])=>(
                        <option key={key} value={key}>{fmt.description||key}</option>
                      ))}
                    </select>
                    {selectedFormat!=='auto'&&dataFormats[selectedFormat]&&(
                      <div style={{fontSize:10,color:C.textMid,lineHeight:1.4,
                        padding:'4px 6px',background:C.bg,borderRadius:4,
                        border:`1px solid ${C.border}`}}>
                        Beklenen: {dataFormats[selectedFormat].columns?.join(', ')}
                      </div>
                    )}
                  </div>
                )}
                <Btn onClick={()=>fileRef.current?.click()} icon={Upload} variant="secondary" size="sm"
                  style={{width:'100%'}} disabled={uploading}>
                  {uploading?'Yükleniyor...':'Veri Yükle (.npy / .csv / .dat)'}
                </Btn>
                <input ref={fileRef} type="file" accept=".npy" style={{display:'none'}}
                  onChange={e=>{uploadFile(e.target.files?.[0]);e.target.value='';}}/>
              </div>

              {[
                {label:'Y — Geometri',prefix:'Y_',sel:selY,setSel:setSelY,name:'dataset_y'},
                {label:'X — Grav/Mag',prefix:'X_',sel:selGM,setSel:setSelGM,name:'dataset_gm'},
                {label:'X — CSAMT',prefix:'x_csamt',sel:selCS,setSel:setSelCS,name:'dataset_cs'},
              ].map(({label,prefix,sel,setSel,name})=>(
                <PanelSection key={name} title={label} icon={Database} defaultOpen={true}>
                  <div style={{display:'flex',flexDirection:'column',gap:4}}>
                    <label style={{display:'flex',alignItems:'center',gap:6,padding:'5px 8px',borderRadius:5,cursor:'pointer',
                                   background:sel===null?`${C.teal}12`:C.bg,border:`1px solid ${sel===null?C.teal+'40':C.border}`}}>
                      <input type="radio" name={name} checked={sel===null} onChange={()=>setSel(null)} style={{accentColor:C.teal}}/>
                      <span style={{fontSize:11,color:C.textMid}}>demo / sentetik</span>
                    </label>
                    {datasets.filter(ds=>prefix.startsWith('x_csamt')?ds.filename.toLowerCase().startsWith('x_csamt'):
                      prefix==='X_'?(!ds.filename.startsWith('Y_')&&!ds.filename.toLowerCase().startsWith('x_csamt')):
                      ds.filename.startsWith('Y_')).map(ds=>(
                      <div key={ds.filename} style={{display:'flex',alignItems:'center',gap:4,padding:'5px 8px',
                                                      borderRadius:5,background:sel===ds.filename?`${C.teal}12`:C.bg,
                                                      border:`1px solid ${sel===ds.filename?C.teal+'40':C.border}`}}>
                        <label style={{display:'flex',alignItems:'center',gap:6,flex:1,cursor:'pointer',minWidth:0}}>
                          <input type="radio" name={name} checked={sel===ds.filename} onChange={()=>setSel(ds.filename)} style={{accentColor:C.teal}}/>
                          <div style={{minWidth:0}}>
                            <div style={{fontSize:11,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{ds.filename}</div>
                            <div style={{fontSize:9,color:C.textLow,fontFamily:'monospace'}}>{ds.shape?.join('×')} · {ds.size_kb}KB</div>
                          </div>
                        </label>
                        <button onClick={()=>deleteDataset(ds.filename)} style={{background:'none',border:'none',cursor:'pointer',color:C.err,padding:2,flexShrink:0}}>
                          <Trash2 size={12}/>
                        </button>
                      </div>
                    ))}
                  </div>
                </PanelSection>
              ))}

              <div style={{padding:12,display:'flex',flexDirection:'column',gap:8}}>
                <Btn onClick={runAnalysis} icon={Play} disabled={loading} style={{width:'100%'}}>
                  {loading?<><Loader2 size={14} style={{animation:'spin 1s linear infinite'}}/>&nbsp;İşleniyor...</>:'Analizi Başlat'}
                </Btn>
                <div style={{display:'flex',gap:6}}>
                  <Btn onClick={saveAnalysis} icon={Save} variant="secondary" size="sm" disabled={saving||!lastRun} style={{flex:1}}>Kaydet</Btn>
                </div>
              </div>

              <PanelSection title="Geçmiş Analizler" icon={History} defaultOpen={false}
                badge={savedAnalyses.length}>
                <div style={{display:'flex',flexDirection:'column',gap:4,maxHeight:200,overflowY:'auto'}}>
                  {savedAnalyses.length===0&&<span style={{fontSize:11,color:C.textLow}}>Kayıtlı analiz yok.</span>}
                  {savedAnalyses.map(a=>(
                    <div key={a.id} style={{display:'flex',alignItems:'center',gap:4,padding:'6px 8px',
                                            borderRadius:5,background:C.bg,border:`1px solid ${C.border}`}}>
                      <button onClick={()=>loadAnalysis(a.id)} style={{background:'none',border:'none',cursor:'pointer',
                                                                         flex:1,textAlign:'left',minWidth:0}}>
                        <div style={{fontSize:11,fontWeight:600,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{a.name}</div>
                        <div style={{fontSize:9,color:C.textLow}}><Tag color={a.type==='joint'?C.purple:C.teal}>{a.type}</Tag></div>
                      </button>
                      <button onClick={async()=>{await apiFetch(`${apiBase}/api/analyses/${a.id}`,{method:'DELETE'});fetchAnalyses();}}
                        style={{background:'none',border:'none',cursor:'pointer',color:C.err,padding:2,flexShrink:0}}>
                        <Trash2 size={12}/>
                      </button>
                    </div>
                  ))}
                </div>
              </PanelSection>
            </>
          )}

          {/* Ters çözüm sekmesi */}
          {leftTab==='ji' && (
            <div style={{padding:12,display:'flex',flexDirection:'column',gap:10}}>
              <div style={{fontSize:11,fontWeight:700,color:C.header,textTransform:'uppercase',
                           letterSpacing:'0.06em',paddingBottom:8,borderBottom:`1px solid ${C.border}`}}>
                Ortak Ters Çözüm (Adam)
              </div>

              {layerConfig.map(({key,label,color})=>(
                <RangeRow key={key} label={`${label} ağırlık`} value={jiWeights[key]}
                  min={0} max={2} step={0.1}
                  onChange={v=>setJiWeights({...jiWeights,[key]:v})}
                  format={v=>v.toFixed(1)}/>
              ))}

              <RangeRow label="İterasyon sayısı" value={jiIter} min={10} max={200} step={5}
                onChange={setJiIter} format={v=>Math.round(v)}/>

              <div style={{marginBottom:4}}>
                <div style={{fontSize:11,color:C.textMid,marginBottom:4}}>Grid çözünürlüğü</div>
                <Select value={jiGridSize} onChange={v=>setJiGridSize(parseInt(v))} options={[
                  {value:8,label:'8³ — çok hızlı'},
                  {value:16,label:'16³ — hızlı (varsayılan)'},
                  {value:21,label:'21³ — native X'},
                  {value:32,label:'32³ — yüksek (yavaş)'},
                ]}/>
              </div>

              <Btn onClick={runJI} icon={jiRunning?Loader2:GitCompare} disabled={jiRunning}
                variant="teal" style={{width:'100%'}}>
                {jiRunning?'Çözülüyor...':'Ters Çözümü Başlat'}
              </Btn>

              <div style={{height:1,background:C.border,margin:'8px 0'}}/>

              <div style={{fontSize:10,color:C.textMid,marginBottom:6}}>
                Ters çözüm sonrası belirsizliği ölç:
              </div>
              <Btn onClick={()=>{setRightTab('uq');}} icon={AlertCircle}
                variant="secondary" size="sm" style={{width:'100%'}}>
                Belirsizlik Analizine Geç →
              </Btn>

              {jiSummary && (
                <div style={{background:`${C.ok}12`,border:`1px solid ${C.ok}40`,borderRadius:6,padding:10}}>
                  <div style={{fontSize:10,color:C.ok,fontWeight:700,marginBottom:4}}>Son Sonuç</div>
                  <div style={{fontSize:11,fontFamily:'monospace',color:C.text,lineHeight:1.7}}>
                    <div>Misfit: {jiSummary.initial?.toFixed(4)} → <strong style={{color:C.ok}}>{jiSummary.final?.toFixed(4)}</strong></div>
                    <div>RMSE: <strong>{jiSummary.rmse?.toFixed(4)}</strong></div>
                    <div style={{fontSize:9,color:C.textLow,marginTop:2}}>{jiSummary.dataset}</div>
                  </div>
                </div>
              )}

              {jiHistory.length > 0 && (
                <div>
                  <div style={{fontSize:11,fontWeight:600,color:C.header,marginBottom:4}}>Yakınsama</div>
                  <ResponsiveContainer width="100%" height={100}>
                    <LineChart data={jiHistory} margin={{top:2,right:4,left:-28,bottom:0}}>
                      <YAxis tick={{fontSize:8,fill:C.textLow}}/>
                      <Tooltip contentStyle={{fontSize:10,background:C.surface,border:`1px solid ${C.border}`}}/>
                      <Line type="monotone" dataKey="misfit" name="Toplam" stroke={C.teal} dot={false} strokeWidth={2}/>
                      {settings.grav&&<Line dataKey="misfit_grav" name="Grav" stroke={C.tealL} dot={false} strokeWidth={1}/>}
                      {settings.mag&&<Line dataKey="misfit_mag" name="Mag" stroke={C.accent} dot={false} strokeWidth={1}/>}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

          {/* Metrik kartlar — her sekmede alt */}
          <div style={{ marginTop:'auto', padding:12, borderTop:`1px solid ${C.border}`,
                        display:'flex', flexDirection:'column', gap:6 }}>
            <MetricCard label="Kütle" value={(metrics.mass/1e6).toFixed(2)} unit="Mt" icon={Box} color={C.teal}/>
            <MetricCard label="Hacim" value={(metrics.volume/1e6).toFixed(2)} unit="Mm³" icon={Layers} color={C.purple}/>
          </div>
        </aside>

        {/* ── Merkez viewer ── */}
        <main style={{ flex:1, position:'relative', minWidth:0, background: viewBg==='dark'?'#0F172A':'#EFF6FF' }}
          ref={viewerRef}>

          {/* Viewer araç çubuğu */}
          <div style={{ position:'absolute', top:10, left:10, zIndex:30, display:'flex', gap:6 }}>
            {viewMode==='slice' && (
              <div style={{ display:'flex', gap:4, background:'rgba(255,255,255,0.95)',
                            border:`1px solid ${C.border}`, borderRadius:6, padding:4,
                            boxShadow:'0 2px 8px rgba(0,0,0,0.1)' }}>
                {['x','y','z'].map(ax=>(
                  <button key={ax} onClick={()=>setSliceAxis(ax)}
                    style={{ padding:'3px 10px', borderRadius:4, border:'none', cursor:'pointer',
                             fontSize:11, fontWeight:700,
                             background: sliceAxis===ax ? C.teal : 'transparent',
                             color: sliceAxis===ax ? '#fff' : C.textMid }}>
                    {ax.toUpperCase()}
                  </button>
                ))}
                <div style={{width:1,background:C.border,margin:'2px 4px'}}/>
                <input type="range" min={0} max={(activeDisplay?.length??16)-1} value={sliceIdx}
                  onChange={e=>{setSliceIdx(parseInt(e.target.value));stopSweep();}}
                  style={{width:80,accentColor:C.teal}}/>
                <span style={{fontSize:10,fontFamily:'monospace',color:C.header,minWidth:20,textAlign:'center'}}>
                  {sliceIdx}
                </span>
              </div>
            )}
          </div>

          <button onClick={toggleFs}
            style={{ position:'absolute', top:10, right:10, zIndex:30,
                     background:'rgba(255,255,255,0.95)', border:`1px solid ${C.border}`,
                     borderRadius:6, padding:6, cursor:'pointer', color: C.header,
                     boxShadow:'0 2px 8px rgba(0,0,0,0.1)' }}>
            {isFs?<Minimize2 size={14}/>:<Maximize2 size={14}/>}
          </button>

          {/* Koordinat bilgisi */}
          <div style={{ position:'absolute', bottom:10, left:10, zIndex:20,
                        background:'rgba(255,255,255,0.9)', border:`1px solid ${C.border}`,
                        borderRadius:4, padding:'4px 8px', fontSize:9, fontFamily:'monospace',
                        color: C.textMid }}>
            Domain: 480×480×480 m · Grid: {activeDisplay?.length??'—'}³
          </div>

          {/* İçerik */}
          {viewMode==='3d' && (
            <>
              <Scene3D modelData={activeDisplay} isoThreshold={isoThreshold}
                opacity={opacity3d} showTopo={showTopo} topoOpacity={topoOpacity} bgColor={viewBg}/>
              {activeDisplay?.length > 0 && (
                <ColorBar vmin={colorRange.min} vmax={colorRange.max} label="Cevher [0–1]"/>
              )}
              {!activeDisplay?.length && (
                <div style={{ position:'absolute', inset:0, display:'flex', flexDirection:'column',
                              alignItems:'center', justifyContent:'center', color: viewBg==='dark'?'#475569':C.textMid }}>
                  <Mountain size={64} style={{opacity:0.2,marginBottom:16}}/>
                  <div style={{fontSize:14, fontWeight:600}}>Analiz çalıştırın</div>
                  <div style={{fontSize:12,marginTop:4,opacity:0.7}}>Sol panelden veri seçip başlatın.</div>
                </div>
              )}
            </>
          )}
          {viewMode==='slice' && (
            <SliceView modelData={activeDisplay} axis={sliceAxis} idx={sliceIdx}
              colorRange={colorRange} sweeping={sweeping}/>
          )}
          {viewMode==='anomaly' && (
            <AnomalyMap modelData={activeDisplay} colorRange={colorRange}/>
          )}
          {viewMode==='stats' && (
            <StatsPanel modelData={activeDisplay} colorRange={colorRange}
              jiHistory={jiHistory} jiCorrelation={jiCorr} jiSummary={jiSummary}/>
          )}
        </main>

        {/* ── Sağ Panel: dikey ikon rail + içerik ── */}
        <aside style={{ display:'flex', borderLeft:`1.5px solid ${C.border}`, flexShrink:0 }}>

          {/* Dikey ikon rail */}
          <div style={{ width:46, background: theme==='dark'?'#0A0C10':'#0D1117',
            borderRight:`1px solid ${C.border}`, display:'flex', flexDirection:'column',
            alignItems:'center', paddingTop:8, gap:1, flexShrink:0 }}>
            {rightTabs.map(({id,label,icon:Icon})=>(
              <button key={id} onClick={()=>setRightTab(id)}
                title={label}
                style={{ width:40, height:40, display:'flex', flexDirection:'column',
                  alignItems:'center', justifyContent:'center', gap:2,
                  border:'none', cursor:'pointer', borderRadius:3,
                  background: rightTab===id ? `${C.accent}20` : 'transparent',
                  color: rightTab===id ? C.accent : 'rgba(255,255,255,0.28)',
                  transition:'all 0.12s',
                  borderRight: rightTab===id ? `2px solid ${C.accent}` : '2px solid transparent' }}>
                <Icon size={14}/>
                <span style={{ fontSize:7, fontWeight:700, letterSpacing:'0.03em',
                  textTransform:'uppercase', lineHeight:1,
                  color: rightTab===id ? C.accent : 'rgba(255,255,255,0.22)' }}>
                  {label.length>5 ? label.slice(0,5) : label}
                </span>
              </button>
            ))}
          </div>

          {/* İçerik alanı */}
          <div style={{ width:256, background: C.panel,
                        display:'flex', flexDirection:'column', flexShrink:0 }}>

          {/* Sekme başlığı */}
          <div style={{ height:36, borderBottom:`1px solid ${C.border}`,
            display:'flex', alignItems:'center', padding:'0 12px',
            flexShrink:0, gap:6 }}>
            {(() => { const t=rightTabs.find(x=>x.id===rightTab); const I=t?.icon;
              return (<>
                {I && <I size={13} color={C.accent}/>}
                <span style={{ fontSize:11, fontWeight:700, color:C.text,
                  letterSpacing:'0.04em', textTransform:'uppercase' }}>
                  {t?.label}
                </span>
              </>); })()}
          </div>

          <div style={{flex:1,overflowY:'auto',minHeight:0}}>
            {rightTab==='simpeg' && (
              <div style={{padding:12,overflowY:'auto',height:'100%'}}>
                <div style={{fontSize:11,fontWeight:700,color:C.header,textTransform:'uppercase',
                             letterSpacing:'0.06em',marginBottom:4,display:'flex',alignItems:'center',gap:6}}>
                  SimPEG Tikhonov Inversion
                  {simpegAvailable===true  && <Tag color={C.ok}>✓ Kurulu</Tag>}
                  {simpegAvailable===false && <Tag color={C.err}>✗ Kurulu Değil</Tag>}
                  {simpegAvailable===null  && <Tag color={C.textLow}>— Kontrol ediliyor</Tag>}
                </div>

                <div style={{fontSize:10,color:C.textMid,marginBottom:12,lineHeight:1.6,
                             background:`${C.purple}08`,border:`1px solid ${C.purple}20`,
                             borderRadius:5,padding:'6px 8px'}}>
                  Kendi Adam solver'ımızdan farkı: analitik sensitivity matrix + Gauss-Newton
                  iterasyonu + otomatik beta ayarı. Daha yavaş ama teorik olarak daha doğru.
                </div>

                {simpegAvailable===false && apiBase!=='http://127.0.0.1:8000' && (
                  <div style={{background:`${C.err}10`,border:`1px solid ${C.err}30`,
                               borderRadius:5,padding:'8px 10px',marginBottom:12,fontSize:11,color:C.err}}>
                    SimPEG kurulu değil. Backend'de çalıştırın:<br/>
                    <code style={{fontFamily:'monospace',fontSize:10}}>pip install simpeg discretize</code>
                  </div>
                )}

                <div style={{marginBottom:10}}>
                  <div style={{fontSize:10,color:C.textMid,marginBottom:4}}>Grid çözünürlüğü</div>
                  <Select value={simpegNbc} onChange={v=>setSimpegNbc(parseInt(v))} options={[
                    {value:8, label:'8³ — hızlı test'},
                    {value:16,label:'16³ — varsayılan'},
                    {value:32,label:'32³ — yüksek (yavaş)'},
                  ]}/>
                </div>

                <RangeRow label="Maksimum iterasyon" value={simpegIter} min={5} max={40} step={1}
                  onChange={setSimpegIter} format={v=>`${Math.round(v)}`}/>
                <RangeRow label="Smallness (α_s)" value={simpegAlphaS} min={-6} max={0} step={0.5}
                  onChange={setSimpegAlphaS} format={v=>`10^${v.toFixed(1)}`}/>
                <RangeRow label="Smoothness (α_x)" value={simpegAlphaX} min={-2} max={2} step={0.5}
                  onChange={setSimpegAlphaX} format={v=>`10^${v.toFixed(1)}`}/>

                <div style={{fontSize:10,color:C.textLow,marginBottom:8,fontStyle:'italic'}}>
                  α_s küçük → pürüzlü/detaylı · α_s büyük → düzgün/basit<br/>
                  α_x büyük → laterale yumuşak · χ-fact=1.0 (hedef misfit)
                </div>

                <Btn onClick={runSimPEG} disabled={simpegRunning||simpegAvailable===false}
                  variant="teal" icon={simpegRunning?Loader2:GitCompare} style={{width:'100%',marginBottom:12}}>
                  {simpegRunning?'SimPEG çalışıyor...':'SimPEG Inversion Başlat'}
                </Btn>

                {simpegResult && (
                  <div style={{background:C.bg,border:`1px solid ${C.border}`,borderRadius:6,padding:10}}>
                    <div style={{fontSize:11,fontWeight:700,color:C.header,marginBottom:8}}>Sonuç</div>
                    <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:6,marginBottom:8}}>
                      {[
                        {l:'Yöntem', v:'Tikhonov L2', c:C.purple},
                        {l:'Grid',   v:`${simpegResult.mesh_info?.nbc}³`, c:C.teal},
                        simpegResult.results?.density_kgm3 && {l:'Δρ max', v:`${simpegResult.results.density_kgm3?.toFixed(0)} kg/m³`, c:C.accent},
                        simpegResult.results?.susceptibility_max && {l:'χ max', v:simpegResult.results.susceptibility_max?.toExponential(2)+' SI', c:C.ok},
                      ].filter(Boolean).map(({l,v,c})=>(
                        <div key={l} style={{background:C.surface,border:`1px solid ${C.border}`,
                                             borderRadius:5,padding:'6px 8px'}}>
                          <div style={{fontSize:9,color:C.textLow}}>{l}</div>
                          <div style={{fontSize:12,fontWeight:700,fontFamily:'monospace',color:c}}>{v}</div>
                        </div>
                      ))}
                    </div>

                    {/* Yakınsama */}
                    {(simpegResult.history?.grav?.length>0 || simpegResult.history?.mag?.length>0) && (
                      <div>
                        <div style={{fontSize:10,fontWeight:600,color:C.header,marginBottom:4}}>
                          Gauss-Newton yakınsaması
                        </div>
                        <ResponsiveContainer width="100%" height={80}>
                          <LineChart margin={{top:2,right:4,left:-24,bottom:0}}>
                            <YAxis tick={{fontSize:8}}/>
                            <Tooltip contentStyle={{fontSize:9}}/>
                            {simpegResult.history?.grav?.length>0 && (
                              <Line data={simpegResult.history.grav.map((v,i)=>({i,f:v}))}
                                dataKey="f" dot={false} stroke={C.teal} strokeWidth={2} name="Grav φ"/>
                            )}
                            {simpegResult.history?.mag?.length>0 && (
                              <Line data={simpegResult.history.mag.map((v,i)=>({i,f:v}))}
                                dataKey="f" dot={false} stroke={C.accent} strokeWidth={2} name="Mag φ"/>
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                        <div style={{fontSize:9,color:C.textLow,marginTop:2}}>
                          φ = φ_d + β·φ_m (data misfit + regularization)
                        </div>
                      </div>
                    )}

                    <div style={{marginTop:8,padding:'6px 8px',background:`${C.teal}08`,
                                 borderRadius:5,fontSize:10,color:C.textMid}}>
                      {simpegResult.method}
                    </div>
                  </div>
                )}
              </div>
            )}
            {rightTab==='uq' && (
              <div style={{padding:12,overflowY:'auto',height:'100%'}}>
                <div style={{fontSize:11,fontWeight:700,color:C.header,textTransform:'uppercase',
                             letterSpacing:'0.06em',marginBottom:4}}>Jeolojik Belirsizlik</div>
                <div style={{fontSize:10,color:C.textMid,marginBottom:12,lineHeight:1.6,
                             background:`${C.teal}08`,border:`1px solid ${C.teal}20`,
                             borderRadius:5,padding:'6px 8px'}}>
                  Farklı başlangıç noktalarından {uqNReal} bağımsız inversion çalıştırır.
                  Sonuçların istatistiksel dağılımı jeolojik belirsizliği ölçer.
                </div>

                <RangeRow label="Realizasyon sayısı" value={uqNReal} min={3} max={15} step={1}
                  onChange={setUqNReal} format={v=>`${Math.round(v)} adet`}/>
                <RangeRow label="Veri gürültüsü (σ)" value={uqNoise} min={0} max={0.15} step={0.01}
                  onChange={setUqNoise} format={v=>`%${(v*100).toFixed(0)}`}/>
                <RangeRow label="İterasyon / realizasyon" value={uqIter} min={10} max={80} step={5}
                  onChange={setUqIter} format={v=>`${Math.round(v)}`}/>

                <div style={{fontSize:10,color:C.textLow,marginBottom:8,fontStyle:'italic'}}>
                  Tahmini süre: ~{Math.round(uqNReal * uqIter * 0.5)} sn (GPU'ya göre değişir)
                </div>

                <Btn onClick={runUQ} disabled={uqRunning} variant="teal" icon={uqRunning?Loader2:Activity}
                  style={{width:'100%',marginBottom:12}}>
                  {uqRunning?`Çalışıyor... (${uqNReal} realizasyon)`:'Belirsizlik Analizi Başlat'}
                </Btn>

                {uqResult && (<>
                  {/* Özet */}
                  <div style={{background:`${C.ok}10`,border:`1px solid ${C.ok}30`,borderRadius:6,
                               padding:10,marginBottom:12}}>
                    <div style={{fontSize:11,fontWeight:700,color:C.ok,marginBottom:6}}>Sonuç Özeti</div>
                    <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:6}}>
                      {[
                        {l:'Yüksek güven', v:`%${uqResult.summary.high_conf_pct}`, c:C.ok,
                         tip:'CV<0.3 — tüm realizasyonlar tutarlı'},
                        {l:'Düşük güven',  v:`%${uqResult.summary.low_conf_pct}`, c:C.err,
                         tip:'CV>0.7 — realizasyonlar çelişiyor'},
                        {l:'Ort. RMSE',    v:uqResult.summary.mean_rmse?.toFixed(4), c:C.teal, tip:''},
                        {l:'RMSE std',     v:uqResult.summary.std_rmse?.toFixed(4),  c:C.purple, tip:''},
                      ].map(({l,v,c,tip})=>(
                        <div key={l} style={{background:C.surface,border:`1px solid ${C.border}`,
                                             borderRadius:5,padding:'6px 8px'}} title={tip}>
                          <div style={{fontSize:9,color:C.textLow}}>{l}</div>
                          <div style={{fontSize:14,fontWeight:700,fontFamily:'monospace',color:c}}>{v}</div>
                        </div>
                      ))}
                    </div>

                    {/* Realizasyon RMSE grafiği */}
                    <div style={{marginTop:8}}>
                      <div style={{fontSize:10,color:C.textMid,marginBottom:4}}>RMSE dağılımı (realizasyonlar)</div>
                      <ResponsiveContainer width="100%" height={60}>
                        <BarChart data={uqResult.summary.rmse_per_real.map((v,i)=>({r:`R${i+1}`,rmse:v}))}
                          margin={{top:0,right:4,left:-24,bottom:0}}>
                          <XAxis dataKey="r" tick={{fontSize:8}}/>
                          <YAxis tick={{fontSize:8}}/>
                          <Tooltip contentStyle={{fontSize:10}}/>
                          <Bar dataKey="rmse" fill={C.teal} radius={[2,2,0,0]}/>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Katman seçici */}
                  <div style={{marginBottom:8}}>
                    <div style={{fontSize:11,fontWeight:700,color:C.header,marginBottom:6,
                                 textTransform:'uppercase',letterSpacing:'0.05em'}}>3D'de Görüntüle</div>
                    <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:4}}>
                      {[
                        {id:'mean', label:'Ortalama',   desc:'En olası model',      color:C.teal},
                        {id:'std',  label:'Std Sapma',  desc:'Belirsizlik haritası', color:C.purple},
                        {id:'cv',   label:'CV (norm.)', desc:'CV>0.5 güvenilmez',   color:C.accent},
                        {id:'p10',  label:'P10 (kötü)', desc:'Kötümser senaryo',     color:C.err},
                        {id:'p90',  label:'P90 (iyi)',  desc:'İyimser senaryo',      color:C.ok},
                      ].map(({id,label,desc,color})=>(
                        <button key={id} onClick={()=>applyUQLayer(id)}
                          style={{padding:'7px 6px',borderRadius:5,cursor:'pointer',border:'none',
                                  background: uqDisplayMode===id ? `${color}18` : C.bg,
                                  outline: uqDisplayMode===id ? `2px solid ${color}` : 'none',
                                  transition:'all 0.15s'}}>
                          <div style={{fontSize:11,fontWeight:700,color}}>{label}</div>
                          <div style={{fontSize:9,color:C.textLow,marginTop:1}}>{desc}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Yakınsama eğrileri */}
                  <div>
                    <div style={{fontSize:11,fontWeight:700,color:C.header,marginBottom:4,
                                 textTransform:'uppercase',letterSpacing:'0.05em'}}>Realizasyon yakınsamaları</div>
                    <ResponsiveContainer width="100%" height={100}>
                      <LineChart margin={{top:2,right:4,left:-24,bottom:0}}>
                        <XAxis type="number" dataKey="iter" tick={{fontSize:8}} hide/>
                        <YAxis tick={{fontSize:8,fill:C.textLow}}/>
                        <Tooltip contentStyle={{fontSize:9}}/>
                        {uqResult.histories.map((h,i)=>(
                          <Line key={i} data={h.convergence.map((v,k)=>({iter:k,misfit:v}))}
                            dataKey="misfit" dot={false} strokeWidth={1.2}
                            stroke={`hsl(${180+i*30},60%,50%)`} name={`R${i+1}`}/>
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{fontSize:9,color:C.textLow,marginTop:4}}>
                      Eğriler birbirine yakınsa → veri model çözümünü iyi kısıtlıyor
                    </div>
                  </div>
                </>)}
              </div>
            )}
            {rightTab==='radio' && (
              <div style={{padding:12,display:'flex',flexDirection:'column',gap:10,overflowY:'auto'}}>
                <div style={{fontSize:12,fontWeight:700,color:C.text,
                  borderBottom:`1px solid ${C.border}`,paddingBottom:6}}>
                  Radyometri & Isı Akışı (U/Th/K)
                </div>
                <div style={{fontSize:11,color:C.textMid,lineHeight:1.5}}>
                  Cevher modelinden U/Th/K konsantrasyonu hesaplar →
                  gammaray sayımı, Th/U alterasyon indeksi, radyojenik ısı akışı.
                </div>

                {/* Petrofizik parametreler */}
                <div style={{background:C.bg,border:`1px solid ${C.border}`,borderRadius:6,padding:10}}>
                  <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:8}}>
                    Petrofizik (Beylikova analogu)
                  </div>
                  {[
                    ['U (arka plan)', 'u_bg',  'ppm', 0.5, 10,  0.5],
                    ['U (cevher)',    'u_ore', 'ppm', 5,   50,  1],
                    ['Th (arka plan)','th_bg', 'ppm', 2,   30,  1],
                    ['Th (cevher)',   'th_ore','ppm', 20,  150, 5],
                    ['K (arka plan)', 'k_bg',  '%',   0.5, 4,   0.1],
                    ['K (cevher)',    'k_ore', '%',   2,   8,   0.1],
                    ['k ısıl iletkenlik','k_thermal','W/mK',1,4,0.1],
                  ].map(([label,key,unit,min,max,step])=>(
                    <div key={key} style={{display:'flex',alignItems:'center',
                      justifyContent:'space-between',marginBottom:5,gap:8}}>
                      <span style={{fontSize:10,color:C.textMid,flex:'0 0 130px'}}>{label}</span>
                      <input type="range" min={min} max={max} step={step}
                        value={radParams[key]}
                        onChange={e=>setRadParams(p=>({...p,[key]:parseFloat(e.target.value)}))}
                        style={{flex:1,accentColor:C.teal}}/>
                      <span style={{fontSize:10,fontFamily:'monospace',
                        color:C.teal,minWidth:55,textAlign:'right'}}>
                        {radParams[key].toFixed(1)} {unit}
                      </span>
                    </div>
                  ))}
                </div>

                <Btn onClick={runRadiometry} icon={radRunning?Loader2:Waves}
                  variant='teal' size='sm' style={{width:'100%'}}
                  disabled={radRunning||!radAvailable}>
                  {radRunning?'Hesaplanıyor...':!radAvailable?'Modül yok':'Radyometri Hesapla'}
                </Btn>

                {!radAvailable&&(
                  <div style={{fontSize:10,color:C.textMid,background:C.panel,
                    borderRadius:4,padding:'8px 10px',border:`1px solid ${C.border}`,lineHeight:1.6}}>
                    <div style={{fontWeight:700,color:C.warn,marginBottom:4}}>
                      Radyometri modülü aktif değil
                    </div>
                    Colab'daki <code style={{fontFamily:'monospace',fontSize:9}}>engines/</code> klasörüne
                    eklemek için:
                    <div style={{marginTop:6,fontFamily:'monospace',fontSize:9,
                      background:C.bg,padding:'5px 8px',borderRadius:3,color:C.teal}}>
                      cp radiometry.py engines/<br/>
                      cp heat_flow_fvm.py engines/
                    </div>
                    <div style={{marginTop:4,color:C.textLow,fontSize:9}}>
                      Sonra Server Başlat hücresini yeniden çalıştır.
                    </div>
                  </div>
                )}

                {radResult&&(
                  <div style={{display:'flex',flexDirection:'column',gap:8}}>
                    {/* REE Hedef İndeksi */}
                    {radResult.ree_index&&(
                      <div style={{background:C.bg,border:`1px solid ${C.border}`,borderRadius:6,padding:10}}>
                        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:6}}>
                          REE Hedef İndeksi
                        </div>
                        {[
                          ['Maks. olasılık', (radResult.ree_index.stats.max_prob*100).toFixed(1)+'%'],
                          ['Yüksek olasılıklı hücre', radResult.ree_index.stats.high_prob_cells+' voksel'],
                          ['Ort. Th/U', radResult.ree_index.stats.Th_U_mean?.toFixed(2)],
                        ].map(([k,v])=>(
                          <div key={k} style={{display:'flex',justifyContent:'space-between',
                            fontSize:11,marginBottom:3}}>
                            <span style={{color:C.textMid}}>{k}</span>
                            <span style={{fontFamily:'monospace',
                              color:radResult.ree_index.stats.max_prob>0.6?'#22c55e':C.teal}}>{v}</span>
                          </div>
                        ))}
                        <div style={{fontSize:10,color:C.textMid,marginTop:6,
                          padding:'4px 8px',borderRadius:4,
                          background:radResult.ree_index.stats.max_prob>0.6?'#22c55e18':'#f9731618',
                          border:`1px solid ${radResult.ree_index.stats.max_prob>0.6?'#22c55e40':'#f9731640'}`}}>
                          {radResult.ree_index.stats.interpretation}
                        </div>
                      </div>
                    )}

                    {/* Radyometri */}
                    {radResult.radiometry&&(
                      <div style={{background:C.bg,border:`1px solid ${C.border}`,borderRadius:6,padding:10}}>
                        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:6}}>
                          Gammaray (yüzey)
                        </div>
                        {[
                          ['TC maks',  radResult.radiometry.stats.TC_max?.toFixed(1)+' cps'],
                          ['TC ort',   radResult.radiometry.stats.TC_mean?.toFixed(1)+' cps'],
                          ['Th/U maks',radResult.radiometry.stats.Th_U_max?.toFixed(2)],
                          ['Doz maks', radResult.radiometry.stats.dose_max?.toFixed(1)+' nGy/h'],
                        ].map(([k,v])=>(
                          <div key={k} style={{display:'flex',justifyContent:'space-between',
                            fontSize:11,marginBottom:3}}>
                            <span style={{color:C.textMid}}>{k}</span>
                            <span style={{fontFamily:'monospace',color:C.teal}}>{v}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Isı Akışı */}
                    {radResult.heat_flow&&(
                      <div style={{background:C.bg,border:`1px solid ${C.border}`,borderRadius:6,padding:10}}>
                        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:6}}>
                          Radyojenik Isı Akışı
                        </div>
                        {[
                          ['Ort. ısı akışı', radResult.heat_flow.stats.heat_flux_mean_mw_m2?.toFixed(1)+' mW/m²'],
                          ['Maks. ısı akışı', radResult.heat_flow.stats.heat_flux_max_mw_m2?.toFixed(1)+' mW/m²'],
                          ['Ort. Q',  radResult.heat_flow.stats.Q_mean_uw_m3?.toFixed(2)+' μW/m³'],
                          ['Maks. Q', radResult.heat_flow.stats.Q_max_uw_m3?.toFixed(2)+' μW/m³'],
                        ].map(([k,v])=>(
                          <div key={k} style={{display:'flex',justifyContent:'space-between',
                            fontSize:11,marginBottom:3}}>
                            <span style={{color:C.textMid}}>{k}</span>
                            <span style={{fontFamily:'monospace',color:C.teal}}>{v}</span>
                          </div>
                        ))}
                        <div style={{fontSize:10,color:C.textMid,marginTop:6,lineHeight:1.4}}>
                          {radResult.heat_flow.stats.interpretation}
                        </div>
                      </div>
                    )}

                    <div style={{fontSize:10,color:C.textMid}}>
                      Dataset: {radResult.dataset_used} · Grid: {radResult.grid_size}³
                    </div>
                  </div>
                )}
              </div>
            )}
            {rightTab==='fvm' && (
              <div style={{padding:12,display:'flex',flexDirection:'column',gap:10,overflowY:'auto'}}>
                <div style={{fontSize:12,fontWeight:700,color:C.text,borderBottom:`1px solid ${C.border}`,paddingBottom:6,marginBottom:2}}>
                  FVM vs Prizma Karşılaştırması
                </div>
                <div style={{fontSize:11,color:C.textMid,lineHeight:1.5}}>
                  Analitik prizma motoru (Nagy/Bhattacharyya) ile Sonlu Hacimler Poisson
                  çözücüsünü aynı model üzerinde karşılaştırır. RMSE ve görsel fark haritası.
                </div>
                <Btn onClick={runFvmCompare} icon={fvmRunning?Loader2:GitCompare}
                  variant='teal' size='sm' style={{width:'100%'}} disabled={fvmRunning||fvmAvailable===false}>
                  {fvmRunning?'Hesaplanıyor...':!fvmAvailable?'FVM modülü yok':'Karşılaştırmayı Başlat'}
                </Btn>
                {fvmAvailable===false && apiBase!=='http://127.0.0.1:8000' && (
                  <div style={{fontSize:10,color:C.textMid,background:C.bg,
                    borderRadius:3,padding:'8px 10px',border:`1px solid ${C.border}`,lineHeight:1.6}}>
                    <div style={{fontWeight:700,color:C.warn,marginBottom:4}}>FVM modülü aktif değil</div>
                    <code style={{fontSize:9,color:C.teal}}>engines/gravity_fvm.py</code> ve{' '}
                    <code style={{fontSize:9,color:C.teal}}>magnetic_fvm.py</code> Colab'daki
                    engines/ klasöründe mevcut — server'ı yeniden başlatın.
                  </div>
                )}
                {fvmResult&&(
                  <div style={{display:'flex',flexDirection:'column',gap:8}}>
                    <div style={{fontSize:11,color:C.textMid,fontFamily:'monospace'}}>
                      Dataset: {fvmResult.dataset_used} · Grid: {fvmResult.grid_size}³
                    </div>
                    {fvmResult.result?.gravity&&(
                      <div style={{background:C.bg,border:`1px solid ${C.border}`,borderRadius:6,padding:10}}>
                        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:6}}>
                          Gravite (mGal)
                        </div>
                        {[
                          ['RMSE', fvmResult.result.gravity.rmse_mgal?.toFixed(4)+' mGal'],
                          ['Maks. sapma', fvmResult.result.gravity.max_diff_mgal?.toFixed(4)+' mGal'],
                          ['Göreli RMSE', fvmResult.result.gravity.rel_rmse_pct?.toFixed(2)+'%'],
                          ['Prizma süresi', fvmResult.result.gravity.time_prism_s?.toFixed(3)+'s'],
                          ['FVM süresi', fvmResult.result.gravity.time_fvm_s?.toFixed(3)+'s'],
                        ].map(([k,v])=>(
                          <div key={k} style={{display:'flex',justifyContent:'space-between',
                            fontSize:11,marginBottom:3}}>
                            <span style={{color:C.textMid}}>{k}</span>
                            <span style={{fontFamily:'monospace',color:C.teal}}>{v}</span>
                          </div>
                        ))}
                        <div style={{fontSize:10,color:C.textMid,marginTop:6,lineHeight:1.4}}>
                          {fvmResult.result.gravity.rel_rmse_pct < 5
                            ? '✓ Motorlar %5 içinde uyumlu'
                            : '⚠ %5 üzerinde sapma — sınır etkisi veya grid çözünürlüğü'}
                        </div>
                      </div>
                    )}
                    {fvmResult.result?.magnetic&&(
                      <div style={{background:C.bg,border:`1px solid ${C.border}`,borderRadius:6,padding:10}}>
                        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:6}}>
                          Manyetik TMI (nT)
                        </div>
                        {[
                          ['RMSE', fvmResult.result.magnetic.rmse_nt?.toFixed(4)+' nT'],
                          ['Maks. sapma', fvmResult.result.magnetic.max_diff_nt?.toFixed(4)+' nT'],
                          ['Göreli RMSE', fvmResult.result.magnetic.rel_rmse_pct?.toFixed(2)+'%'],
                          ['Prizma süresi', fvmResult.result.magnetic.time_prism_s?.toFixed(3)+'s'],
                          ['FVM süresi', fvmResult.result.magnetic.time_fvm_s?.toFixed(3)+'s'],
                        ].map(([k,v])=>(
                          <div key={k} style={{display:'flex',justifyContent:'space-between',
                            fontSize:11,marginBottom:3}}>
                            <span style={{color:C.textMid}}>{k}</span>
                            <span style={{fontFamily:'monospace',color:C.teal}}>{v}</span>
                          </div>
                        ))}
                        <div style={{fontSize:10,color:C.textMid,marginTop:6,lineHeight:1.4}}>
                          {fvmResult.result.magnetic.rel_rmse_pct < 5
                            ? '✓ Motorlar %5 içinde uyumlu'
                            : '⚠ %5 üzerinde sapma'}
                        </div>
                      </div>
                    )}
                    <div style={{fontSize:10,color:C.textMid,lineHeight:1.5,
                      borderTop:`1px solid ${C.border}`,paddingTop:8}}>
                      {fvmResult.note}
                    </div>
                  </div>
                )}
              </div>
            )}
            {rightTab==='geom' && (
              <div style={{padding:12,overflowY:'auto'}}>
                <GeometryPanel
                  onGenerated={(d, filename)=>{
                    updateModel(d.model_data);
                    setLastRun('physics');
                    setRightTab('stats');
                    if (filename) {
                      setSelY(filename);
                      fetchDatasets();
                    }
                  }}
                  log={log}
                  apiBase={apiBase}
                />
              </div>
            )}
            {rightTab==='stats' && (
              <StatsPanel modelData={activeDisplay} colorRange={colorRange}
                jiHistory={jiHistory} jiCorrelation={jiCorr} jiSummary={jiSummary}/>
            )}
            {rightTab==='filter' && (
              <div style={{padding:12}}>
                <div style={{fontSize:11,fontWeight:700,color:C.header,textTransform:'uppercase',
                             letterSpacing:'0.06em',marginBottom:10}}>Ham Veri Filtreleme</div>
                <FilterPanel modelData={modelData} onFiltered={data=>{
                  setFilteredData(data);
                  const flat=data.flat(2);
                  setColorRange({min:Math.min(...flat),max:Math.max(...flat)});
                  log('ok','Filtre uygulandı.');
                }}/>
                {filteredData && (
                  <div style={{marginTop:8}}>
                    <Btn onClick={()=>{setFilteredData(null);const flat=modelData.flat(2);
                      setColorRange({min:Math.min(...flat),max:Math.max(...flat)});}}
                      icon={RefreshCw} variant="ghost" size="sm" style={{width:'100%',color:C.err}}>
                      Filtreyi Sıfırla
                    </Btn>
                  </div>
                )}
              </div>
            )}
            {rightTab==='export' && (
              <div style={{padding:12}}>
                <div style={{fontSize:11,fontWeight:700,color:C.header,textTransform:'uppercase',
                             letterSpacing:'0.06em',marginBottom:10}}>Dışa Aktarma</div>
                <ExportPanel modelData={activeDisplay} colorRange={colorRange} logs={logs}/>
              </div>
            )}
          </div>

          {/* Log konsolu */}
          <div style={{ height:180, borderTop:`1px solid ${C.border}`, display:'flex', flexDirection:'column' }}>
            <div style={{ padding:'5px 10px', background: C.header, display:'flex', alignItems:'center', gap:5 }}>
              <AlertCircle size={11} color={C.accentL}/>
              <span style={{fontSize:10,fontWeight:700,color:'rgba(255,255,255,0.8)',
                            textTransform:'uppercase',letterSpacing:'0.06em'}}>Log</span>
            </div>
            <div style={{ flex:1, overflowY:'auto', background:'#0F172A', padding:'6px 8px',
                          fontFamily:'monospace', fontSize:10, lineHeight:1.6 }}>
              {logs.map((l,i)=>(
                <div key={i} style={{ display:'flex', gap:6 }}>
                  <span style={{color:'#475569',flexShrink:0}}>{l.t}</span>
                  <span style={{color:l.level==='err'?'#FC8181':l.level==='ok'?'#68D391':'#A0AEC0'}}>
                    {l.msg}
                  </span>
                </div>
              ))}
            </div>
          </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
