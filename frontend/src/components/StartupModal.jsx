import React, { useState } from 'react';
import {
  Brain,
  Layers,
  Ruler,
  Sparkles,
  Calculator,
  ArrowRight,
  CheckCircle2,
  X,
  Play,
  Upload,
  BookOpen,
  ShieldCheck
} from 'lucide-react';

export default function StartupModal({
  isOpen,
  onClose,
  onLaunchDemo,
  onSelectUpload
}) {
  const [dontShowAgain, setDontShowAgain] = useState(false);

  if (!isOpen) return null;

  const handleClose = () => {
    if (dontShowAgain) {
      localStorage.setItem('neuro_ai_hide_welcome', 'true');
    }
    onClose();
  };

  const handleDemoClick = (caseType) => {
    if (dontShowAgain) {
      localStorage.setItem('neuro_ai_hide_welcome', 'true');
    }
    onLaunchDemo(caseType);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="relative w-full max-w-4xl bg-slate-900 border border-slate-700/80 rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Glow ambient background effects */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        {/* Modal Header */}
        <div className="relative p-6 sm:p-8 border-b border-slate-800 flex items-start justify-between bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950">
          <div className="flex items-center space-x-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-indigo-600 via-blue-500 to-cyan-400 p-0.5 shadow-xl shadow-indigo-500/25 flex items-center justify-center">
              <div className="w-full h-full bg-slate-950 rounded-[14px] flex items-center justify-center">
                <Brain className="w-8 h-8 text-cyan-400 animate-pulse" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight">
                  NeuroMultimodal AI
                </h1>
                <span className="text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-indigo-500/20 border border-indigo-500/40 text-indigo-300">
                  v2.5 WHO CNS5
                </span>
                <span className="text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 hidden sm:inline-flex items-center space-x-1">
                  <ShieldCheck className="w-3 h-3" />
                  <span>95.75% Test Acc</span>
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-1">
                Clinical Neuro-Oncology Cockpit combining Deep CNN, Graph Attention (GAT), and WHO CNS5 RAG.
              </p>
            </div>
          </div>

          <button
            onClick={handleClose}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body / Feature Showcase Grid */}
        <div className="p-6 sm:p-8 space-y-6 overflow-y-auto custom-scrollbar flex-1">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Feature 1 */}
            <div className="p-4 rounded-2xl bg-slate-950/70 border border-slate-800 hover:border-indigo-500/50 transition space-y-2">
              <div className="w-8 h-8 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center">
                <Layers className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-bold text-white">Tri-Modal Cross-Attention Fusion</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Fuses 2D spatial features (ResNet18), 7x7 superpixel tissue graphs (GATv2), and molecular biomarkers (IDH1, MGMT, 1p/19q) yielding <strong>95.75% accuracy</strong> and <strong>100% tumor detection</strong>.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="p-4 rounded-2xl bg-slate-950/70 border border-slate-800 hover:border-cyan-500/50 transition space-y-2">
              <div className="w-8 h-8 rounded-xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
                <Ruler className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-bold text-white">Digital Calipers & 4-Compartment Segmentation</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Interactive two-point calipers for sub-millimeter lesion measuring, 3D ellipsoid volume calculation, and automated segmentation of Enhancing Tumor Core, Edema, and Necrosis.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="p-4 rounded-2xl bg-slate-950/70 border border-slate-800 hover:border-amber-500/50 transition space-y-2">
              <div className="w-8 h-8 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center">
                <Sparkles className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-bold text-white">WHO CNS5 Grounded RAG Advisor</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Evidence-based neuropathology consultation reports synthesized by local Ollama models (<code className="text-amber-300">llama3.2</code>, <code className="text-amber-300">deepseek-r1</code>) querying a local ChromaDB vector store.
              </p>
            </div>

            {/* Feature 4 */}
            <div className="p-4 rounded-2xl bg-slate-950/70 border border-slate-800 hover:border-purple-500/50 transition space-y-2">
              <div className="w-8 h-8 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center">
                <Calculator className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-bold text-white">RTOG RPA & Prognostic Stratification</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                RTOG Recursive Partitioning Analysis (Classes III–VI) for Glioblastoma, KPS to ECOG functional converter, and molecular hazard risk scoring.
              </p>
            </div>
          </div>

          {/* Quick Launch Pathways */}
          <div className="p-5 rounded-2xl bg-gradient-to-r from-indigo-950/50 via-slate-950 to-cyan-950/50 border border-indigo-500/30 space-y-3">
            <div className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center space-x-1.5">
              <Play className="w-3.5 h-3.5 text-indigo-400" />
              <span>Quick-Start Demonstrations &amp; Patient Cases</span>
            </div>
            <p className="text-xs text-slate-300">
              Select a pre-configured clinical profile to immediately run the full multimodal pipeline:
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
              <button
                onClick={() => handleDemoClick('case-gbm-01')}
                className="p-3 rounded-xl bg-slate-900/90 border border-red-500/40 hover:bg-red-950/40 transition text-left space-y-1 cursor-pointer group"
              >
                <div className="flex items-center justify-between text-xs font-bold text-red-300">
                  <span>Glioblastoma (GBM)</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition" />
                </div>
                <div className="text-[10px] text-slate-400">WHO Grade IV • IDH1 Wildtype • KPS 75</div>
              </button>

              <button
                onClick={() => handleDemoClick('case-mening-01')}
                className="p-3 rounded-xl bg-slate-900/90 border border-amber-500/40 hover:bg-amber-950/40 transition text-left space-y-1 cursor-pointer group"
              >
                <div className="flex items-center justify-between text-xs font-bold text-amber-300">
                  <span>Meningioma</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition" />
                </div>
                <div className="text-[10px] text-slate-400">Convexity • Dural Tail • KPS 90</div>
              </button>

              <button
                onClick={() => handleDemoClick('case-norm-01')}
                className="p-3 rounded-xl bg-slate-900/90 border border-emerald-500/40 hover:bg-emerald-950/40 transition text-left space-y-1 cursor-pointer group"
              >
                <div className="flex items-center justify-between text-xs font-bold text-emerald-300">
                  <span>Healthy Brain</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition" />
                </div>
                <div className="text-[10px] text-slate-400">Non-neoplastic Control • KPS 100</div>
              </button>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 sm:p-6 border-t border-slate-800 bg-slate-950 flex flex-wrap items-center justify-between gap-3">
          <label className="flex items-center space-x-2 text-xs text-slate-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={dontShowAgain}
              onChange={(e) => setDontShowAgain(e.target.checked)}
              className="rounded bg-slate-900 border-slate-700 text-indigo-600 focus:ring-indigo-500"
            />
            <span>Don't show this welcome screen again on startup</span>
          </label>

          <div className="flex items-center space-x-2.5">
            <button
              onClick={() => {
                handleClose();
                onSelectUpload?.();
              }}
              className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center space-x-1.5 transition cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Upload Custom MRI</span>
            </button>

            <button
              onClick={handleClose}
              className="px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition cursor-pointer flex items-center space-x-1.5"
            >
              <span>Enter Workstation</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
