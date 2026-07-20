import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  Play, Maximize2, Minimize2, Layers, Box, Activity,
  Waves, Magnet, Radio, SlidersHorizontal, Terminal, Gauge,
  Database, Upload, Trash2, GitCompare, ChevronDown, ChevronUp, Loader2,
  Save, History, FolderOpen
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { MarchingCubes } from 'three/examples/jsm/objects/MarchingCubes.js';

function trilinearUpsample(data, factor) {
  const n = data.length;
  const m = n * factor;
  const out = [];
  for (let i = 0; i < m; i++) {
    const gi = i / factor;
    const i0 = Math.min(Math.floor(gi), n - 1);
    const i1 = Math.min(i0 + 1, n - 1);
    const fi = gi - i0;
    out.push([]);
    for (let j = 0; j < m; j++) {
      const gj = j / factor;
      const j0 = Math.min(Math.floor(gj), n - 1);
      const j1 = Math.min(j0 + 1, n - 1);
      const fj = gj - j0;
      out[i].push([]);
      for (let k = 0; k < m; k++) {
        const gk = k / factor;
        const k0 = Math.min(Math.floor(gk), n - 1);
        const k1 = Math.min(k0 + 1, n - 1);
        const fk = gk - k0;
        const c000 = data[i0][j0][k0], c100 = data[i1][j0][k0];
        const c010 = data[i0][j1][k0], c110 = data[i1][j1][k0];
        const c001 = data[i0][j0][k1], c101 = data[i1][j0][k1];
        const c011 = data[i0][j1][k1], c111 = data[i1][j1][k1];
        const c00 = c000 * (1 - fi) + c100 * fi;
        const c10 = c010 * (1 - fi) + c110 * fi;
        const c01 = c001 * (1 - fi) + c101 * fi;
        const c11 = c011 * (1 - fi) + c111 * fi;
        const c0 = c00 * (1 - fj) + c10 * fj;
        const c1 = c01 * (1 - fj) + c11 * fj;
        out[i][j].push(c0 * (1 - fk) + c1 * fk);
      }
    }
  }
  return out;
}

function percentile(sortedArr, p) {
  if (!sortedArr.length) return 0;
  const idx = Math.min(sortedArr.length - 1, Math.max(0, Math.floor(sortedArr.length * p)));
  return sortedArr[idx];
}

function IsosurfaceMesh({ modelData, targetFraction = 0.28 }) {
  const resolution = 32;

  const effect = useMemo(() => {
    const material = new THREE.MeshStandardMaterial({
      color: 0x2dd4bf,
      roughness: 0.5,
      metalness: 0.08,
      flatShading: false,
    });
    const mc = new MarchingCubes(resolution, material, false, false, 100000);
    mc.scale.set(8, 8, 8);
    return mc;
  }, []);

  useEffect(() => {
    if (!modelData || !modelData.length) return;
    const factor = Math.max(1, Math.round(resolution / modelData.length));
    const fine = trilinearUpsample(modelData, factor);
    const n = fine.length;
    const size = resolution;
    const field = effect.field;
    field.fill(0);
    const flatVals = [];
    for (let x = 0; x < size; x++) {
      const fx = Math.min(Math.floor((x / size) * n), n - 1);
      for (let y = 0; y < size; y++) {
        const fy = Math.min(Math.floor((y / size) * n), n - 1);
        for (let z = 0; z < size; z++) {
          const fz = Math.min(Math.floor((z / size) * n), n - 1);
          const v = fine[fx][fy][fz];
          field[x + y * size + z * size * size] = v;
          flatVals.push(v);
        }
      }
    }
    // Sabit bir eşik (örn. 0.15) yalnızca sentetik demo verisinin ölçeğine uyuyordu.
    // Gerçek veri setleri çok farklı değer aralıklarına sahip olabiliyor; eşik veri
    // dağılımının ortasına denk gelirse marching cubes her hücrede içeri/dışarı
    // savrulup "süngerimsi/delikli" bir yüzey üretiyordu. Bunun yerine eşiği,
    // hacmin yaklaşık targetFraction kadarını kapsayacak şekilde veriden hesaplıyoruz.
    flatVals.sort((a, b) => a - b);
    effect.isolation = percentile(flatVals, 1 - targetFraction);
    effect.update();
  }, [modelData, effect, targetFraction]);

  return <primitive object={effect} />;
}

function VoxelScene({ modelData }) {
  return (
    <Canvas camera={{ position: [22, 18, 22], fov: 45 }}>
      <ambientLight intensity={0.75} />
      <directionalLight position={[15, 20, 10]} intensity={1.1} />
      <directionalLight position={[-15, -10, -10]} intensity={0.35} />
      <IsosurfaceMesh modelData={modelData} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
      <gridHelper args={[40, 20, '#94a3b8', '#334155']} position={[0, -12, 0]} />
    </Canvas>
  );
}

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [modelData, setModelData] = useState(null);
  const [results, setResults] = useState({});
  const [metrics, setMetrics] = useState({ mass: 0, volume: 0, heat: 0 });
  const [settings, setSettings] = useState({
    grav: true, mag: true, csamt: false, index: 0
  });
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState([
    { t: '00:00:00', level: 'ok', msg: 'Sistem hazır. Veri yolu tanımlı.' }
  ]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const viewerRef = useRef(null);

  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  const [jiOpen, setJiOpen] = useState(false);
  const [jiRunning, setJiRunning] = useState(false);
  const [jiWeights, setJiWeights] = useState({ grav: 1.0, mag: 1.0, csamt: 1.0 });
  const [jiIter, setJiIter] = useState(24);
  const [jiHistory, setJiHistory] = useState([]);
  const [jiCorrelation, setJiCorrelation] = useState({});
  const [jiSummary, setJiSummary] = useState(null);

  const [lastRunType, setLastRunType] = useState(null); // 'physics' | 'joint' | null
  const [historyOpen, setHistoryOpen] = useState(false);
  const [savedAnalyses, setSavedAnalyses] = useState([]);
  const [saving, setSaving] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const layerConfig = [
    { key: 'grav', label: 'Gravite', sub: 'Bouguer anomali', icon: Gauge },
    { key: 'mag', label: 'Manyetik', sub: 'Toplam alan', icon: Magnet },
    { key: 'csamt', label: 'CSAMT', sub: 'Direnç / faz', icon: Radio },
  ];

  const corrLabels = {
    grav_mag: 'Gravite ↔ Manyetik',
    grav_csamt: 'Gravite ↔ CSAMT',
    mag_csamt: 'Manyetik ↔ CSAMT',
  };

  const timestamp = () => new Date().toLocaleTimeString('tr-TR', { hour12: false });

  const pushLog = (level, msg) => {
    setLogs(prev => [{ t: timestamp(), level, msg }, ...prev].slice(0, 200));
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/data/list`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setDatasets(data.files || []);
    } catch (e) {
      pushLog('err', `Veri seti listesi alınamadı: ${e.message}`);
    }
  };

  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      pushLog('ok', `Backend bağlantısı kuruldu (v${data.version}).`);
    } catch (e) {
      pushLog('err', `Backend'e ulaşılamadı: ${e.message}. Sunucu çalışıyor mu kontrol edin.`);
    }
  };

  // Uygulama açılışında backend sağlık kontrolü — önceden burada tanımlıydı
  // ama hiçbir yerden çağrılmıyordu, bu yüzden bağlantı hataları startup'ta
  // fark edilmiyordu.
  useEffect(() => { checkBackendHealth(); }, []);

  useEffect(() => { fetchDatasets(); }, []);

  const fetchAnalyses = async () => {
    setLoadingHistory(true);
    try {
      const res = await fetch(`${API_BASE}/api/analyses`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSavedAnalyses(data.analyses || []);
    } catch (e) {
      pushLog('err', `Geçmiş analizler alınamadı: ${e.message}`);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => { fetchAnalyses(); }, []);

  const handleUploadClick = () => fileInputRef.current?.click();

  const handleUploadChange = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    if (!file.name.endsWith('.npy')) {
      pushLog('err', 'Sadece .npy dosyaları yüklenebilir.');
      return;
    }
    setUploading(true);
    pushLog('info', `Yükleniyor: ${file.name}`);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${API_BASE}/api/data/upload`, { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      pushLog('ok', `Yüklendi: ${data.filename} (${(data.shape || []).join('×')})`);
      await fetchDatasets();
      setSelectedDataset(data.filename);
    } catch (e) {
      pushLog('err', `Yükleme hatası: ${e.message}`);
    } finally {
      setUploading(false);
    }
  };

  const deleteDataset = async (filename) => {
    try {
      const res = await fetch(`${API_BASE}/api/data/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      pushLog('ok', `Silindi: ${filename}`);
      if (selectedDataset === filename) setSelectedDataset(null);
      await fetchDatasets();
    } catch (e) {
      pushLog('err', `Silme hatası: ${e.message}`);
    }
  };

  const calculateMetrics = (data) => {
    const voxelVol = Math.pow(30, 3);
    let vol = 0, mass = 0, sumVal = 0;
    data.flat(2).forEach(val => {
      if (val > 0.1) {
        vol += voxelVol;
        mass += (val * 2000 * voxelVol);
        sumVal += val;
      }
    });
    return { volume: vol, mass: mass / 1000, heat: sumVal * 0.05 };
  };

  const runAnalysis = async () => {
    setLoading(true);
    pushLog('info', 'Analiz başlatıldı — fizik motoru çağrılıyor...');
    try {
      const response = await fetch(`${API_BASE}/api/run-physics-engine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grav_active: settings.grav,
          mag_active: settings.mag,
          csamt_active: settings.csamt,
          selected_index: settings.index,
          dataset: selectedDataset
        })
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const safeResults = Array.isArray(data.results)
        ? Object.fromEntries(data.results.map((v, i) => [`ch${i}`, v]))
        : (data.results || {});
      setModelData(data.model_data);
      setResults(safeResults);
      setMetrics(calculateMetrics(data.model_data));
      setLastRunType('physics');
      pushLog('ok', `Analiz tamamlandı (${data.dataset_used}) — ${Object.keys(safeResults).length} kanal döndü.`);
    } catch (e) {
      pushLog('err', `HATA: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const runJointInversion = async () => {
    const activeCount = [settings.grav, settings.mag, settings.csamt].filter(Boolean).length;
    if (activeCount < 1) {
      pushLog('err', 'Joint inversion için en az bir katman aktif olmalı.');
      return;
    }
    setJiRunning(true);
    pushLog('info', `Joint inversion başlatıldı (${jiIter} iterasyon)...`);
    try {
      const response = await fetch(`${API_BASE}/api/joint-inversion`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grav_active: settings.grav,
          mag_active: settings.mag,
          csamt_active: settings.csamt,
          selected_index: settings.index,
          dataset: selectedDataset,
          n_iter: jiIter,
          weights: jiWeights,
        })
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      setJiHistory(data.history || []);
      setJiCorrelation(data.correlation || {});
      setJiSummary({
        initial: data.initial_misfit,
        final: data.final_misfit,
        rmse: data.rmse_vs_true_model,
        dataset: data.dataset_used,
      });
      setModelData(data.model_data);
      setMetrics(calculateMetrics(data.model_data));
      setLastRunType('joint');
      pushLog('ok', `Joint inversion bitti — misfit ${data.initial_misfit?.toFixed(4)} → ${data.final_misfit?.toFixed(4)}`);
    } catch (e) {
      pushLog('err', `Joint inversion hatası: ${e.message}`);
    } finally {
      setJiRunning(false);
    }
  };

  const saveCurrentAnalysis = async () => {
    if (!lastRunType) {
      pushLog('err', 'Önce bir analiz çalıştırmalısın.');
      return;
    }
    const defaultName = `${lastRunType === 'joint' ? 'Joint Inversion' : 'Analiz'} — ${new Date().toLocaleString('tr-TR')}`;
    const name = window.prompt('Analiz için bir isim gir:', defaultName);
    if (!name) return; // kullanıcı iptal etti

    setSaving(true);
    try {
      const payload = {
        name,
        type: lastRunType,
        dataset_used: selectedDataset,
        settings,
        results: lastRunType === 'physics' ? results : {},
        metrics,
        model_data: modelData,
        history: lastRunType === 'joint' ? jiHistory : null,
        correlation: lastRunType === 'joint' ? jiCorrelation : null,
        summary: lastRunType === 'joint' ? jiSummary : null,
      };
      const res = await fetch(`${API_BASE}/api/analyses`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      pushLog('ok', `Analiz kaydedildi: "${name}"`);
      await fetchAnalyses();
    } catch (e) {
      pushLog('err', `Kaydetme hatası: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  const loadAnalysis = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/api/analyses/${id}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const record = await res.json();

      if (record.model_data) {
        setModelData(record.model_data);
        setMetrics(record.metrics || calculateMetrics(record.model_data));
      }
      if (record.type === 'physics') {
        setResults(record.results || {});
        setJiHistory([]);
        setJiCorrelation({});
        setJiSummary(null);
      } else if (record.type === 'joint') {
        setJiHistory(record.history || []);
        setJiCorrelation(record.correlation || {});
        setJiSummary(record.summary || null);
        setJiOpen(true);
      }
      if (record.settings) setSettings(record.settings);
      setLastRunType(record.type);
      pushLog('ok', `Yüklendi: "${record.name}"`);
    } catch (e) {
      pushLog('err', `Analiz yüklenemedi: ${e.message}`);
    }
  };

  const deleteAnalysisRecord = async (id, name) => {
    try {
      const res = await fetch(`${API_BASE}/api/analyses/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      pushLog('ok', `Silindi: "${name}"`);
      await fetchAnalyses();
    } catch (e) {
      pushLog('err', `Silme hatası: ${e.message}`);
    }
  };

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
      requestAnimationFrame(() => {
        setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
      });
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      viewerRef.current?.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };

  const activeLayerCount = layerConfig.filter(l => settings[l.key]).length;
  const hasModelData = !!(modelData && modelData.length);



  return (
    <div className="flex flex-col h-screen bg-slate-50 text-slate-800 overflow-hidden font-sans">
      <div className="h-11 flex items-center justify-between px-4 bg-white border-b border-slate-200 shrink-0">
        <div className="flex items-center gap-2.5">
          <Box size={16} className="text-teal-600" strokeWidth={2.5} />
          <span className="text-[13px] font-semibold tracking-wide text-slate-800">GeoPINN</span>
          <span className="text-[13px] font-mono text-teal-600/80">3.0</span>
          <span className="ml-2 text-[10px] uppercase tracking-wider text-slate-500 border border-slate-300 rounded px-1.5 py-0.5">Applied Geophysics Suite</span>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono text-slate-500">
          <span className={`flex items-center gap-1.5 ${loading || jiRunning ? 'text-amber-600' : 'text-teal-600'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${loading || jiRunning ? 'bg-amber-500 animate-pulse' : 'bg-teal-500'}`} />
            {loading ? 'İşleniyor' : jiRunning ? 'Ters çözüm çalışıyor' : 'Hazır'}
          </span>
          <span className="text-slate-300">|</span>
          <span>{activeLayerCount} katman aktif</span>
          <span className="text-slate-300">|</span>
          <span>{selectedDataset || 'demo/sentetik'}</span>
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        <div className="w-72 bg-white border-r border-slate-200 p-4 flex flex-col gap-6 overflow-y-auto">
          <div>
            <h2 className="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-2.5 flex items-center gap-1.5">
              <Layers size={11} /> Veri Katmanları
            </h2>
            <div className="space-y-1.5">
              {layerConfig.map(({ key, label, sub, icon: Icon }) => (
                <label
                  key={key}
                  className={`flex items-center gap-3 p-2.5 rounded-md border cursor-pointer transition-colors ${
                    settings[key]
                      ? 'bg-teal-950/30 border-teal-800/50'
                      : 'bg-slate-100 border-slate-200 hover:border-slate-400'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={settings[key]}
                    onChange={(e) => setSettings({ ...settings, [key]: e.target.checked })}
                    className="accent-teal-500 w-3.5 h-3.5"
                  />
                  <Icon size={15} className={settings[key] ? 'text-teal-400' : 'text-slate-600'} />
                  <div className="flex flex-col leading-tight">
                    <span className="text-[12.5px] font-medium text-slate-800">{label}</span>
                    <span className="text-[10.5px] text-slate-500">{sub}</span>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div>
            <h2 className="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-2.5 flex items-center gap-1.5">
              <SlidersHorizontal size={11} /> Kesit Parametreleri
            </h2>
            <div className="bg-slate-100 border border-slate-200 rounded-md p-3 space-y-1.5">
              <div className="flex items-baseline justify-between">
                <span className="text-[11px] text-slate-500">Kesit İndeksi</span>
                <span className="text-[13px] font-mono text-teal-400">{settings.index}</span>
              </div>
              <input
                type="range" min="0" max="1999" value={settings.index}
                onChange={(e) => setSettings({ ...settings, index: parseInt(e.target.value) })}
                className="w-full accent-teal-500"
              />
              <div className="flex justify-between text-[9.5px] font-mono text-slate-600">
                <span>0</span><span>1999</span>
              </div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2.5">
              <h2 className="text-[10px] uppercase tracking-wider text-slate-500 font-bold flex items-center gap-1.5">
                <Database size={11} /> Veri Seti
              </h2>
              <button
                onClick={handleUploadClick}
                disabled={uploading}
                className="text-[10px] flex items-center gap-1 text-teal-400 hover:text-teal-300 disabled:text-slate-600"
              >
                {uploading ? <Loader2 size={11} className="animate-spin" /> : <Upload size={11} />}
                Yükle
              </button>
              <input ref={fileInputRef} type="file" accept=".npy" className="hidden" onChange={handleUploadChange} />
            </div>
            <div className="space-y-1.5">
              <label
                className={`flex items-center gap-2.5 p-2 rounded-md border cursor-pointer transition-colors ${
                  selectedDataset === null
                    ? 'bg-teal-950/30 border-teal-800/50'
                    : 'bg-slate-100 border-slate-200 hover:border-slate-400'
                }`}
              >
                <input
                  type="radio" name="dataset" checked={selectedDataset === null}
                  onChange={() => setSelectedDataset(null)} className="accent-teal-500 w-3 h-3"
                />
                <span className="text-[11.5px] text-slate-700 flex-1">demo / sentetik</span>
              </label>
              {datasets.length === 0 && (
                <p className="text-[10.5px] text-slate-600 px-1">Henüz veri yüklenmedi.</p>
              )}
              {datasets.map((ds) => (
                <div
                  key={ds.filename}
                  className={`flex items-center gap-2 p-2 rounded-md border transition-colors ${
                    selectedDataset === ds.filename
                      ? 'bg-teal-950/30 border-teal-800/50'
                      : 'bg-slate-100 border-slate-200 hover:border-slate-400'
                  }`}
                >
                  <label className="flex items-center gap-2.5 flex-1 min-w-0 cursor-pointer">
                    <input
                      type="radio" name="dataset" checked={selectedDataset === ds.filename}
                      onChange={() => setSelectedDataset(ds.filename)} className="accent-teal-500 w-3 h-3 shrink-0"
                    />
                    <div className="flex flex-col leading-tight min-w-0">
                      <span className="text-[11.5px] text-slate-700 truncate">{ds.filename}</span>
                      <span className="text-[9.5px] font-mono text-slate-600">
                        {ds.shape ? ds.shape.join('×') : '?'} · {ds.size_kb} KB
                      </span>
                    </div>
                  </label>
                  <button onClick={() => deleteDataset(ds.filename)} className="text-slate-600 hover:text-rose-400 shrink-0 p-1">
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={runAnalysis}
            disabled={loading}
            className="w-full bg-teal-600 hover:bg-teal-500 disabled:bg-slate-700 disabled:cursor-not-allowed transition-colors py-2.5 rounded-md font-semibold text-[13px] flex items-center justify-center gap-2 text-white"
          >
            {loading ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                İşleniyor...
              </>
            ) : (
              <>
                <Play size={14} fill="currentColor" /> Analizi Başlat
              </>
            )}
          </button>

          <button
            onClick={() => setJiOpen(o => !o)}
            className={`w-full py-2 rounded-md font-medium text-[12.5px] flex items-center justify-center gap-2 border transition-colors ${
              jiOpen ? 'bg-indigo-950/40 border-indigo-800/60 text-indigo-300' : 'bg-slate-100 border-slate-200 text-slate-600 hover:border-slate-400'
            }`}
          >
            <GitCompare size={13} /> Joint Inversion Paneli
            {jiOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>

          <button
            onClick={saveCurrentAnalysis}
            disabled={saving || !lastRunType}
            className="w-full py-2 rounded-md font-medium text-[12.5px] flex items-center justify-center gap-2 border border-slate-200 bg-slate-100 text-slate-600 hover:border-teal-700/60 hover:text-teal-400 disabled:opacity-30 disabled:hover:border-slate-200 disabled:hover:text-slate-600 transition-colors"
          >
            {saving ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
            Analizi Kaydet
          </button>

          <div>
            <button
              onClick={() => setHistoryOpen(o => !o)}
              className={`w-full py-2 rounded-md font-medium text-[12.5px] flex items-center justify-center gap-2 border transition-colors ${
                historyOpen ? 'bg-teal-950/30 border-teal-800/50 text-teal-300' : 'bg-slate-100 border-slate-200 text-slate-600 hover:border-slate-400'
              }`}
            >
              <History size={13} /> Geçmiş Analizler
              {historyOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>

            {historyOpen && (
              <div className="mt-2 space-y-1.5 max-h-64 overflow-y-auto">
                {loadingHistory && (
                  <p className="text-[10.5px] text-slate-600 px-1 flex items-center gap-1.5">
                    <Loader2 size={11} className="animate-spin" /> Yükleniyor...
                  </p>
                )}
                {!loadingHistory && savedAnalyses.length === 0 && (
                  <p className="text-[10.5px] text-slate-600 px-1">Henüz kayıtlı analiz yok.</p>
                )}
                {savedAnalyses.map((a) => (
                  <div
                    key={a.id}
                    className="flex items-center gap-2 p-2 rounded-md border border-slate-200 bg-slate-100 hover:border-slate-400 transition-colors"
                  >
                    <button
                      onClick={() => loadAnalysis(a.id)}
                      className="flex items-center gap-2 flex-1 min-w-0 text-left"
                    >
                      <FolderOpen size={13} className="text-slate-600 shrink-0" />
                      <div className="flex flex-col leading-tight min-w-0">
                        <span className="text-[11.5px] text-slate-700 truncate">{a.name}</span>
                        <span className="text-[9.5px] font-mono text-slate-600">
                          {a.type === 'joint' ? 'Joint Inv.' : 'Analiz'} · {a.dataset_used || 'demo'} · {new Date(a.created_at).toLocaleDateString('tr-TR')}
                        </span>
                      </div>
                    </button>
                    <button
                      onClick={() => deleteAnalysisRecord(a.id, a.name)}
                      className="text-slate-600 hover:text-rose-400 shrink-0 p-1"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 relative bg-black min-h-0" ref={viewerRef}>
            <div className="pointer-events-none absolute inset-4 z-10 hidden md:block">
              <div className="absolute top-0 left-0 w-6 h-6 border-t border-l border-teal-800/40" />
              <div className="absolute top-0 right-0 w-6 h-6 border-t border-r border-teal-800/40" />
              <div className="absolute bottom-0 left-0 w-6 h-6 border-b border-l border-teal-800/40" />
              <div className="absolute bottom-0 right-0 w-6 h-6 border-b border-r border-teal-800/40" />
            </div>
            <div className="absolute top-4 left-4 z-20 text-[10px] font-mono text-slate-600 uppercase tracking-wider">
              İç Yapı Görünümü
            </div>
            <button
              onClick={toggleFullscreen}
              className="absolute top-4 right-4 z-20 bg-slate-100 border border-slate-300 p-2 rounded-md hover:border-teal-700/60 hover:text-teal-400 transition-colors"
            >
              {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
            {hasModelData ? (
              <VoxelScene modelData={modelData} />
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-slate-600 gap-4">
                <Layers size={56} className="opacity-30" strokeWidth={1.2} />
                <p className="text-[13px] text-slate-500">Henüz model verisi yok — bir analiz çalıştırın.</p>
              </div>
            )}
          </div>

          {jiOpen && (
            <div className="h-72 bg-white border-t border-slate-200 flex shrink-0 overflow-hidden">
              <div className="w-64 border-r border-slate-200 p-3.5 flex flex-col gap-3 overflow-y-auto shrink-0">
                <h3 className="text-[10px] uppercase tracking-wider text-indigo-400 font-bold flex items-center gap-1.5">
                  <GitCompare size={11} /> Ortak Ters Çözüm (Gradyan/Adam)
                </h3>
                {['grav', 'mag', 'csamt'].map((key) => (
                  <div key={key} className="space-y-1">
                    <div className="flex items-baseline justify-between">
                      <span className="text-[10.5px] text-slate-500">{layerConfig.find(l => l.key === key)?.label || key.toUpperCase()} ağırlık</span>
                      <span className="text-[11px] font-mono text-indigo-400">{jiWeights[key].toFixed(1)}</span>
                    </div>
                    <input
                      type="range" min="0" max="2" step="0.1" value={jiWeights[key]}
                      onChange={(e) => setJiWeights({ ...jiWeights, [key]: parseFloat(e.target.value) })}
                      disabled={!settings[key]}
                      className="w-full accent-indigo-500 disabled:opacity-30"
                    />
                  </div>
                ))}
                <div className="space-y-1">
                  <div className="flex items-baseline justify-between">
                    <span className="text-[10.5px] text-slate-500">İterasyon</span>
                    <span className="text-[11px] font-mono text-indigo-400">{jiIter}</span>
                  </div>
                  <input
                    type="range" min="6" max="150" step="1" value={jiIter}
                    onChange={(e) => setJiIter(parseInt(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                </div>
                <button
                  onClick={runJointInversion}
                  disabled={jiRunning}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 transition-colors py-2 rounded-md font-semibold text-[12.5px] flex items-center justify-center gap-2 text-white mt-1"
                >
                  {jiRunning ? (
                    <><Loader2 size={13} className="animate-spin" /> Çözülüyor...</>
                  ) : (
                    <><Play size={12} fill="currentColor" /> Çalıştır</>
                  )}
                </button>
                {jiSummary && (
                  <div className="text-[10px] font-mono text-slate-500 space-y-0.5 pt-1 border-t border-slate-200 mt-1">
                    <div>Misfit: {jiSummary.initial?.toFixed(4)} → <span className="text-teal-400">{jiSummary.final?.toFixed(4)}</span></div>
                    <div>RMSE: {jiSummary.rmse?.toFixed(4)}</div>
                  </div>
                )}
              </div>
              <div className="flex-1 p-3.5 min-w-0">
                <h3 className="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1.5">Yakınsama (Misfit)</h3>
                {jiHistory.length > 0 ? (
                  <ResponsiveContainer width="100%" height="88%">
                    <LineChart data={jiHistory} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="iter" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'iterasyon', position: 'insideBottom', offset: -2, fill: '#475569', fontSize: 10 }} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', fontSize: 11 }} labelStyle={{ color: '#334155' }} />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      <Line type="monotone" dataKey="misfit" name="Toplam" stroke="#2dd4bf" dot={false} strokeWidth={2} />
                      {settings.grav && <Line type="monotone" dataKey="misfit_grav" name="Gravite" stroke="#60a5fa" dot={false} strokeWidth={1.3} />}
                      {settings.mag && <Line type="monotone" dataKey="misfit_mag" name="Manyetik" stroke="#f59e0b" dot={false} strokeWidth={1.3} />}
                      {settings.csamt && <Line type="monotone" dataKey="misfit_csamt" name="CSAMT" stroke="#f472b6" dot={false} strokeWidth={1.3} />}
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-[11.5px] text-slate-700">Henüz çalıştırılmadı.</div>
                )}
              </div>
              <div className="w-56 border-l border-slate-200 p-3.5 shrink-0 overflow-y-auto">
                <h3 className="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-2">Korelasyon (Pearson)</h3>
                {Object.keys(jiCorrelation).length > 0 ? (
                  <table className="w-full text-[11px]">
                    <tbody>
                      {Object.entries(jiCorrelation).map(([key, val]) => (
                        <tr key={key} className="border-b border-slate-200 last:border-0">
                          <td className="py-1.5 pr-2 text-slate-600">{corrLabels[key] || key}</td>
                          <td className={`py-1.5 text-right font-mono ${Math.abs(val) > 0.6 ? 'text-teal-400' : Math.abs(val) > 0.3 ? 'text-amber-400' : 'text-slate-500'}`}>
                            {val.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="text-[11px] text-slate-700">Veri yok.</p>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="w-80 bg-white border-l border-slate-200 p-4 flex flex-col gap-5 overflow-hidden">
          <h2 className="text-[10px] uppercase tracking-wider text-slate-500 font-bold flex items-center gap-1.5">
            <Activity size={11} /> Jeofiziksel Özet
          </h2>
          <div className="space-y-2">
            <div className="p-3.5 bg-slate-100 rounded-md border border-slate-200 flex items-center justify-between">
              <div>
                <div className="text-slate-500 text-[10.5px] uppercase tracking-wide">Toplam Kütle</div>
                <div className="text-lg font-mono text-teal-400 mt-0.5">{metrics.mass.toFixed(2)}<span className="text-[11px] text-slate-500 ml-1">Ton</span></div>
              </div>
              <Box size={18} className="text-teal-800" />
            </div>
            <div className="p-3.5 bg-slate-100 rounded-md border border-slate-200 flex items-center justify-between">
              <div>
                <div className="text-slate-500 text-[10.5px] uppercase tracking-wide">Toplam Hacim</div>
                <div className="text-lg font-mono text-teal-400 mt-0.5">{metrics.volume.toFixed(0)}<span className="text-[11px] text-slate-500 ml-1">m³</span></div>
              </div>
              <Layers size={18} className="text-teal-800" />
            </div>
            <div className="p-3.5 bg-slate-100 rounded-md border border-slate-200 flex items-center justify-between">
              <div>
                <div className="text-slate-500 text-[10.5px] uppercase tracking-wide">Isı İndeksi</div>
                <div className="text-lg font-mono text-amber-400 mt-0.5">{metrics.heat.toFixed(4)}<span className="text-[11px] text-slate-500 ml-1">W/m²</span></div>
              </div>
              <Waves size={18} className="text-amber-900" />
            </div>
          </div>
          <div className="flex-1 min-h-0 flex flex-col gap-2">
            <h3 className="text-[10px] uppercase tracking-wider text-slate-500 font-bold flex items-center gap-1.5">
              <Terminal size={11} /> Log Konsolu
            </h3>
            <div className="flex-1 bg-slate-900 rounded-md border border-slate-200 p-2.5 overflow-y-auto font-mono text-[11px] leading-relaxed">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-slate-500 shrink-0">{log.t}</span>
                  <span className={
                    log.level === 'err' ? 'text-rose-400' :
                    log.level === 'ok' ? 'text-teal-400' : 'text-slate-400'
                  }>{log.msg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}