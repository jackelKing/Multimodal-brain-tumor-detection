import React, { useState } from 'react';
import { Eye, Layers, Network, Sliders, Maximize2, RotateCcw, SplitSquareVertical } from 'lucide-react';

export default function MriViewer({ originalImage, camOverlay, gnnOverlay, results }) {
  const [viewMode, setViewMode] = useState('cam'); // 'orig', 'cam', 'gnn', 'split'
  const [splitPos, setSplitPos] = useState(50);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);

  const resetFilters = () => {
    setBrightness(100);
    setContrast(100);
    setSplitPos(50);
  };

  const currentImageSrc = () => {
    if (viewMode === 'orig') return originalImage;
    if (viewMode === 'gnn') return gnnOverlay || originalImage;
    return camOverlay || originalImage;
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
      {/* Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Eye className="w-5 h-5 text-cyan-400" />
            <span>Interactive Neuroimaging Visualizer</span>
          </h2>
          <p className="text-xs text-slate-400">Multi-modal layer overlays with Grad-CAM & Graph Topology.</p>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-medium">
          <button
            onClick={() => setViewMode('orig')}
            className={`px-3 py-1.5 rounded-lg transition ${
              viewMode === 'orig' ? 'bg-slate-800 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Raw MRI
          </button>
          <button
            onClick={() => setViewMode('cam')}
            className={`px-3 py-1.5 rounded-lg transition flex items-center space-x-1 ${
              viewMode === 'cam' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>CNN Grad-CAM</span>
          </button>
          <button
            onClick={() => setViewMode('gnn')}
            className={`px-3 py-1.5 rounded-lg transition flex items-center space-x-1 ${
              viewMode === 'gnn' ? 'bg-cyan-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Network className="w-3.5 h-3.5" />
            <span>GNN Graph</span>
          </button>
          <button
            onClick={() => setViewMode('split')}
            className={`px-3 py-1.5 rounded-lg transition flex items-center space-x-1 ${
              viewMode === 'split' ? 'bg-purple-600 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <SplitSquareVertical className="w-3.5 h-3.5" />
            <span>Split Comparison</span>
          </button>
        </div>
      </div>

      {/* Main Canvas Display */}
      <div className="relative aspect-square max-h-[420px] mx-auto rounded-xl overflow-hidden border border-slate-800 bg-black flex items-center justify-center select-none">
        {originalImage ? (
          viewMode === 'split' ? (
            /* Split Comparison Mode */
            <div className="relative w-full h-full">
              {/* Underlying Original Image */}
              <img
                src={originalImage}
                alt="Original MRI"
                className="absolute inset-0 w-full h-full object-contain"
                style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
              />
              {/* Overlay Image Clipped */}
              <div
                className="absolute inset-0 overflow-hidden"
                style={{ width: `${splitPos}%`, borderRight: '2px solid #818CF8' }}
              >
                <img
                  src={camOverlay || originalImage}
                  alt="Grad-CAM Overlay"
                  className="absolute inset-0 w-full h-full object-contain max-w-none"
                  style={{
                    width: '100%',
                    height: '100%',
                    filter: `brightness(${brightness}%) contrast(${contrast}%)`
                  }}
                />
                <span className="absolute top-3 left-3 px-2 py-1 bg-indigo-950/80 border border-indigo-500/50 text-[10px] text-indigo-300 font-bold rounded">
                  CNN Grad-CAM
                </span>
              </div>
              <span className="absolute top-3 right-3 px-2 py-1 bg-slate-900/80 border border-slate-700 text-[10px] text-slate-300 font-bold rounded">
                Raw MRI
              </span>
              {/* Slider thumb controller */}
              <input
                type="range"
                min="0"
                max="100"
                value={splitPos}
                onChange={(e) => setSplitPos(Number(e.target.value))}
                className="absolute inset-x-0 bottom-4 w-4/5 mx-auto opacity-70 hover:opacity-100 z-10 accent-indigo-500"
              />
            </div>
          ) : (
            /* Standard Single View Mode */
            <div className="relative w-full h-full flex items-center justify-center">
              <img
                src={currentImageSrc()}
                alt="MRI Scan View"
                className="w-full h-full object-contain transition-all"
                style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
              />
              {/* Mode Badge in corner */}
              <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider bg-slate-900/80 backdrop-blur border border-slate-700 text-slate-200">
                {viewMode === 'orig' && 'Original T1CE/T2 MRI'}
                {viewMode === 'cam' && 'CNN Class Activation Mapping (Grad-CAM)'}
                {viewMode === 'gnn' && 'GNN 7x7 Superpixel Adjacency Graph'}
              </div>
            </div>
          )
        ) : (
          <div className="text-center text-slate-500 space-y-2">
            <Eye className="w-8 h-8 mx-auto opacity-40" />
            <p className="text-xs">Select a clinical case or upload an MRI scan to preview.</p>
          </div>
        )}
      </div>

      {/* Image Adjustments Bar */}
      <div className="flex items-center justify-between gap-4 pt-2 border-t border-slate-800/80 text-xs text-slate-400">
        <div className="flex items-center space-x-4 flex-1">
          <div className="flex items-center space-x-2 flex-1 max-w-[180px]">
            <span>Brightness:</span>
            <input
              type="range"
              min="50"
              max="150"
              value={brightness}
              onChange={(e) => setBrightness(Number(e.target.value))}
              className="w-full accent-cyan-500 h-1 bg-slate-800 rounded"
            />
          </div>
          <div className="flex items-center space-x-2 flex-1 max-w-[180px]">
            <span>Contrast:</span>
            <input
              type="range"
              min="50"
              max="150"
              value={contrast}
              onChange={(e) => setContrast(Number(e.target.value))}
              className="w-full accent-cyan-500 h-1 bg-slate-800 rounded"
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

      {/* Explainability Summaries */}
      {results && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 text-xs">
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
