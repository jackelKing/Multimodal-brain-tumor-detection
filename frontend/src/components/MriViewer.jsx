import React, { useState, useRef } from 'react';
import {
  Eye,
  Layers,
  Network,
  SplitSquareVertical,
  Activity,
  Ruler,
  RotateCcw,
  Sliders,
  CheckCircle2,
  Trash2,
  Info
} from 'lucide-react';

export default function MriViewer({
  originalImage,
  camOverlay,
  gnnOverlay,
  segOverlay,
  results
}) {
  const [viewMode, setViewMode] = useState('cam'); // 'orig', 'cam', 'gnn', 'seg', 'split'
  const [splitPos, setSplitPos] = useState(50);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [overlayOpacity, setOverlayOpacity] = useState(85);

  // Caliper measurement tool state
  const [caliperActive, setCaliperActive] = useState(false);
  const [caliperPoints, setCaliperPoints] = useState([]); // [{x: %, y: %}, ...]
  const [caliperMeasurement, setCaliperMeasurement] = useState(null);

  const imageContainerRef = useRef(null);

  const resetFilters = () => {
    setBrightness(100);
    setContrast(100);
    setOverlayOpacity(85);
    setSplitPos(50);
  };

  const clearCaliper = () => {
    setCaliperPoints([]);
    setCaliperMeasurement(null);
  };

  // Canvas click handler for caliper tool
  const handleCanvasClick = (e) => {
    if (!caliperActive || !imageContainerRef.current) return;

    const rect = imageContainerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;

    if (caliperPoints.length === 0 || caliperPoints.length === 2) {
      setCaliperPoints([{ x, y }]);
      setCaliperMeasurement(null);
    } else if (caliperPoints.length === 1) {
      const p1 = caliperPoints[0];
      const p2 = { x, y };
      setCaliperPoints([p1, p2]);

      // Calculate distance in mm assuming 224x224 and 0.9375 mm/pixel
      const pixelSpacing = 0.9375;
      const dx = ((p2.x - p1.x) / 100) * 224 * pixelSpacing;
      const dy = ((p2.y - p1.y) / 100) * 224 * pixelSpacing;
      const distMm = Math.sqrt(dx * dx + dy * dy);

      // Volume estimation assuming spherical/ellipsoid approximation V = (pi/6) * d^3
      const volumeCm3 = ((Math.PI / 6.0) * Math.pow(distMm, 3)) / 1000.0;

      setCaliperMeasurement({
        distMm: distMm.toFixed(1),
        volumeCm3: volumeCm3.toFixed(2),
        midX: (p1.x + p2.x) / 2,
        midY: (p1.y + p2.y) / 2
      });
    }
  };

  const getOverlayImageSrc = () => {
    if (viewMode === 'gnn') return gnnOverlay || originalImage;
    if (viewMode === 'seg') return segOverlay || originalImage;
    return camOverlay || originalImage;
  };

  const vm = results?.volumetric_metrics;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
      {/* Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Eye className="w-5 h-5 text-cyan-400" />
            <span>Multi-Modal Neuroimaging Suite</span>
          </h2>
          <p className="text-xs text-slate-400">
            Grad-CAM, 7x7 GNN topology, 4-compartment segmentation &amp; digital calipers.
          </p>
        </div>

        {/* View Mode Switcher */}
        <div className="flex flex-wrap items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-medium">
          <button
            onClick={() => setViewMode('orig')}
            className={`px-2.5 py-1.5 rounded-lg transition ${
              viewMode === 'orig' ? 'bg-slate-800 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Raw MRI
          </button>
          <button
            onClick={() => setViewMode('cam')}
            className={`px-2.5 py-1.5 rounded-lg transition flex items-center space-x-1 ${
              viewMode === 'cam' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Grad-CAM</span>
          </button>
          <button
            onClick={() => setViewMode('gnn')}
            className={`px-2.5 py-1.5 rounded-lg transition flex items-center space-x-1 ${
              viewMode === 'gnn' ? 'bg-cyan-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Network className="w-3.5 h-3.5" />
            <span>GNN Graph</span>
          </button>
          <button
            onClick={() => setViewMode('seg')}
            className={`px-2.5 py-1.5 rounded-lg transition flex items-center space-x-1 ${
              viewMode === 'seg' ? 'bg-emerald-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Segmentation</span>
          </button>
          <button
            onClick={() => setViewMode('split')}
            className={`px-2.5 py-1.5 rounded-lg transition flex items-center space-x-1 ${
              viewMode === 'split' ? 'bg-purple-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <SplitSquareVertical className="w-3.5 h-3.5" />
            <span>Split View</span>
          </button>
        </div>
      </div>

      {/* Main Canvas Display */}
      <div
        ref={imageContainerRef}
        onClick={handleCanvasClick}
        className={`relative aspect-square max-h-[440px] mx-auto rounded-xl overflow-hidden border border-slate-800 bg-black flex items-center justify-center select-none ${
          caliperActive ? 'cursor-crosshair' : 'cursor-default'
        }`}
      >
        {originalImage ? (
          viewMode === 'split' ? (
            /* Split Comparison Mode */
            <div className="relative w-full h-full">
              <img
                src={originalImage}
                alt="Original MRI"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
              />
              <div
                className="absolute inset-0 overflow-hidden"
                style={{ width: `${splitPos}%`, borderRight: '2px solid #818CF8' }}
              >
                <img
                  src={getOverlayImageSrc()}
                  alt="Overlay"
                  className="absolute inset-0 w-full h-full object-contain max-w-none pointer-events-none"
                  style={{
                    width: '100%',
                    height: '100%',
                    filter: `brightness(${brightness}%) contrast(${contrast}%)`
                  }}
                />
                <span className="absolute top-3 left-3 px-2 py-1 bg-indigo-950/90 border border-indigo-500/50 text-[10px] text-indigo-300 font-bold rounded shadow">
                  Overlay Layer
                </span>
              </div>
              <span className="absolute top-3 right-3 px-2 py-1 bg-slate-900/90 border border-slate-700 text-[10px] text-slate-300 font-bold rounded shadow">
                Raw MRI
              </span>
              <input
                type="range"
                min="0"
                max="100"
                value={splitPos}
                onChange={(e) => setSplitPos(Number(e.target.value))}
                className="absolute inset-x-0 bottom-4 w-4/5 mx-auto opacity-75 hover:opacity-100 z-10 accent-indigo-500"
              />
            </div>
          ) : (
            /* Standard View Mode with Opacity Blending */
            <div className="relative w-full h-full flex items-center justify-center">
              {/* Underlying Raw Anatomy */}
              <img
                src={originalImage}
                alt="Raw MRI Anatomy"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
              />

              {/* Overlaid Layer */}
              {viewMode !== 'orig' && (
                <img
                  src={getOverlayImageSrc()}
                  alt="Diagnostic Overlay"
                  className="absolute inset-0 w-full h-full object-contain transition-opacity duration-200 pointer-events-none"
                  style={{
                    opacity: overlayOpacity / 100,
                    filter: `brightness(${brightness}%) contrast(${contrast}%)`
                  }}
                />
              )}

              {/* View Badge */}
              <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider bg-slate-900/85 backdrop-blur border border-slate-700 text-slate-200 shadow">
                {viewMode === 'orig' && 'Original T1-CE / T2 MRI'}
                {viewMode === 'cam' && 'CNN Spatial Grad-CAM Focus'}
                {viewMode === 'gnn' && 'GNN 7x7 Superpixel Adjacency Graph'}
                {viewMode === 'seg' && '4-Compartment Morphological Segmentation'}
              </div>
            </div>
          )
        ) : (
          <div className="text-center text-slate-500 space-y-2">
            <Eye className="w-8 h-8 mx-auto opacity-40" />
            <p className="text-xs">Select a clinical case or upload an MRI scan to preview.</p>
          </div>
        )}

        {/* SVG Caliper Overlay */}
        {caliperPoints.length > 0 && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-20">
            {caliperPoints[0] && (
              <g>
                <circle
                  cx={`${caliperPoints[0].x}%`}
                  cy={`${caliperPoints[0].y}%`}
                  r="6"
                  fill="#06B6D4"
                  stroke="#FFFFFF"
                  strokeWidth="2"
                  className="animate-pulse"
                />
                <text
                  x={`${caliperPoints[0].x + 2}%`}
                  y={`${caliperPoints[0].y - 2}%`}
                  fill="#06B6D4"
                  fontSize="12"
                  fontWeight="bold"
                >
                  A
                </text>
              </g>
            )}

            {caliperPoints[1] && (
              <g>
                <line
                  x1={`${caliperPoints[0].x}%`}
                  y1={`${caliperPoints[0].y}%`}
                  x2={`${caliperPoints[1].x}%`}
                  y2={`${caliperPoints[1].y}%`}
                  stroke="#06B6D4"
                  strokeWidth="2.5"
                  strokeDasharray="4 4"
                />
                <circle
                  cx={`${caliperPoints[1].x}%`}
                  cy={`${caliperPoints[1].y}%`}
                  r="6"
                  fill="#06B6D4"
                  stroke="#FFFFFF"
                  strokeWidth="2"
                />
                <text
                  x={`${caliperPoints[1].x + 2}%`}
                  y={`${caliperPoints[1].y - 2}%`}
                  fill="#06B6D4"
                  fontSize="12"
                  fontWeight="bold"
                >
                  B
                </text>
              </g>
            )}
          </svg>
        )}

        {/* Caliper Measurement Badge */}
        {caliperMeasurement && (
          <div
            className="absolute z-30 px-3 py-1.5 rounded-lg bg-slate-950/95 border border-cyan-500 shadow-xl text-center pointer-events-none"
            style={{
              left: `${Math.min(Math.max(caliperMeasurement.midX, 20), 80)}%`,
              top: `${Math.min(Math.max(caliperMeasurement.midY - 12, 10), 80)}%`,
              transform: 'translate(-50%, -50%)'
            }}
          >
            <div className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider">Caliper Diameter</div>
            <div className="text-sm font-mono font-extrabold text-white">
              {caliperMeasurement.distMm} <span className="text-xs text-slate-400">mm</span>
            </div>
            <div className="text-[10px] text-slate-400 font-mono">
              Vol: <span className="text-cyan-300 font-bold">{caliperMeasurement.volumeCm3} cm³</span>
            </div>
          </div>
        )}
      </div>

      {/* Segmentation Legend Bar (when in seg mode) */}
      {viewMode === 'seg' && (
        <div className="flex flex-wrap items-center justify-center gap-3 p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-xs">
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded-full bg-red-500 shadow-sm" />
            <span className="text-slate-300 font-medium">Enhancing Tumor (ET)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded-full bg-amber-400 shadow-sm" />
            <span className="text-slate-300 font-medium">Peritumoral Edema (ED)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded-full bg-blue-500 shadow-sm" />
            <span className="text-slate-300 font-medium">Necrotic Core (NCR)</span>
          </div>
        </div>
      )}

      {/* Adjustments & Caliper Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800 text-xs text-slate-400">
        <div className="flex flex-wrap items-center gap-4">
          {/* Caliper Toggle */}
          <button
            onClick={() => {
              setCaliperActive(!caliperActive);
              if (caliperActive) clearCaliper();
            }}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border font-semibold transition ${
              caliperActive
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500'
                : 'bg-slate-950 text-slate-300 border-slate-800 hover:border-slate-700'
            }`}
          >
            <Ruler className="w-3.5 h-3.5 text-cyan-400" />
            <span>{caliperActive ? 'Caliper Active (Click A → B)' : 'Digital Caliper'}</span>
          </button>

          {caliperActive && caliperPoints.length > 0 && (
            <button
              onClick={clearCaliper}
              className="flex items-center space-x-1 px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition text-[11px]"
            >
              <Trash2 className="w-3 h-3 text-red-400" />
              <span>Clear</span>
            </button>
          )}

          {/* Opacity slider */}
          {viewMode !== 'orig' && viewMode !== 'split' && (
            <div className="flex items-center space-x-2">
              <span>Overlay:</span>
              <input
                type="range"
                min="20"
                max="100"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                className="w-20 accent-cyan-500 h-1 bg-slate-800 rounded"
              />
              <span className="text-[10px] font-mono text-slate-500">{overlayOpacity}%</span>
            </div>
          )}

          {/* Brightness */}
          <div className="flex items-center space-x-2">
            <span>Bri:</span>
            <input
              type="range"
              min="50"
              max="150"
              value={brightness}
              onChange={(e) => setBrightness(Number(e.target.value))}
              className="w-16 accent-indigo-500 h-1 bg-slate-800 rounded"
            />
          </div>

          {/* Contrast */}
          <div className="flex items-center space-x-2">
            <span>Con:</span>
            <input
              type="range"
              min="50"
              max="150"
              value={contrast}
              onChange={(e) => setContrast(Number(e.target.value))}
              className="w-16 accent-indigo-500 h-1 bg-slate-800 rounded"
            />
          </div>
        </div>

        <button
          onClick={resetFilters}
          className="flex items-center space-x-1 px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 transition cursor-pointer"
        >
          <RotateCcw className="w-3 h-3" />
          <span>Reset</span>
        </button>
      </div>

      {/* Volumetric Caliper KPI Summary Bar */}
      {vm && (
        <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800/90 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-300 flex items-center space-x-1.5">
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
              <span>Quantitative Volumetric Delineation</span>
            </span>
            <span className="text-[10px] text-slate-500 font-mono">0.9375 mm/px DICOM Spacing</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
            <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-[10px] text-slate-400">Major Caliper</div>
              <div className="font-mono font-bold text-white mt-0.5">
                {vm.major_diameter_mm ? vm.major_diameter_mm.toFixed(1) : '—'} mm
              </div>
            </div>
            <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-[10px] text-slate-400">Minor Caliper</div>
              <div className="font-mono font-bold text-white mt-0.5">
                {vm.minor_diameter_mm ? vm.minor_diameter_mm.toFixed(1) : '—'} mm
              </div>
            </div>
            <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-[10px] text-slate-400">2D Area</div>
              <div className="font-mono font-bold text-white mt-0.5">
                {vm.area_mm2 ? vm.area_mm2.toFixed(0) : '—'} mm²
              </div>
            </div>
            <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-[10px] text-slate-400">3D Volume</div>
              <div className="font-mono font-bold text-emerald-400 mt-0.5">
                {vm.estimated_volume_cm3 ? vm.estimated_volume_cm3.toFixed(2) : '—'} cm³
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Explainability Summaries */}
      {results && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1 text-xs">
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
            <div className="flex items-center space-x-1.5 text-indigo-400 font-semibold">
              <Layers className="w-3.5 h-3.5" />
              <span>CNN Spatial Localization</span>
            </div>
            <p className="text-slate-300 leading-relaxed text-[11px]">{results.gradcam_summary}</p>
          </div>

          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
            <div className="flex items-center space-x-1.5 text-cyan-400 font-semibold">
              <Network className="w-3.5 h-3.5" />
              <span>GNN Tissue Topology</span>
            </div>
            <p className="text-slate-300 leading-relaxed text-[11px]">{results.gnn_summary}</p>
          </div>
        </div>
      )}
    </div>
  );
}
