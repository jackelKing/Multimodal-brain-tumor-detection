import React from 'react';
import {
  Brain,
  Database,
  Cpu,
  Sparkles,
  BarChart3,
  HelpCircle,
  PlayCircle,
  FileDown,
  ShieldCheck,
  BookOpen
} from 'lucide-react';

export default function Navbar({
  systemStatus,
  onOpenAnalytics,
  onOpenStartupTour,
  onOpenClinicalGuide,
  onExportDossier,
  hasResults,
  activeTab
}) {
  const isOnline = systemStatus?.status === 'online';
  const activeModel = systemStatus?.active_model || 'llama3.2:latest';
  const chunks = systemStatus?.chroma_chunks || 20;

  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur-md sticky top-0 z-40 px-3 sm:px-6 py-2.5">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
        {/* Brand */}
        <div className="flex items-center space-x-2.5 sm:space-x-3 cursor-pointer select-none" onClick={onOpenStartupTour}>
          <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-blue-500 to-cyan-400 p-0.5 shadow-lg shadow-indigo-500/20 flex items-center justify-center shrink-0">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Brain className="w-5 h-5 sm:w-6 sm:h-6 text-cyan-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-1.5 sm:space-x-2">
              <span className="text-sm sm:text-lg font-black tracking-tight text-white">
                NeuroMultimodal AI
              </span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-500/20 border border-indigo-500/40 text-indigo-300">
                v2.5
              </span>
              <span className="hidden xl:inline-flex items-center space-x-1 text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span>95.75% Acc</span>
              </span>
            </div>
            <p className="text-[10px] sm:text-[11px] text-slate-400 hidden sm:block">
              WHO CNS5 Neuro-Oncology Diagnostic &amp; Decision Support Suite
            </p>
          </div>
        </div>

        {/* Action Controls & Navigation */}
        <div className="flex items-center space-x-1.5 sm:space-x-2 text-xs">
          {/* Clinical Reference Guide Button (Always visible on all screens!) */}
          <button
            onClick={onOpenClinicalGuide}
            className={`flex items-center space-x-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg border font-medium transition cursor-pointer ${
              activeTab === 'guide'
                ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300 font-bold'
                : 'bg-slate-800/80 hover:bg-slate-700/80 border-slate-700 text-slate-200'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <span>Clinical Guide</span>
          </button>

          {/* Quick Tour Button */}
          <button
            onClick={onOpenStartupTour}
            className="flex items-center space-x-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg bg-gradient-to-r from-indigo-600/30 to-cyan-600/30 hover:from-indigo-600/40 hover:to-cyan-600/40 border border-indigo-500/40 text-indigo-200 font-semibold transition cursor-pointer"
          >
            <PlayCircle className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden sm:inline">Tour</span>
          </button>

          {/* Export Complete Dossier Button */}
          {hasResults && (
            <button
              onClick={onExportDossier}
              className="flex items-center space-x-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/40 text-emerald-300 font-semibold transition cursor-pointer"
            >
              <FileDown className="w-3.5 h-3.5 text-emerald-400" />
              <span className="hidden lg:inline">Dossier</span>
            </button>
          )}

          {/* Ablation Analytics */}
          <button
            onClick={onOpenAnalytics}
            className="flex items-center space-x-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-medium transition cursor-pointer"
          >
            <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
            <span className="hidden sm:inline">Analytics</span>
          </button>

          {/* Live Ollama Status Badge */}
          <div className="hidden lg:flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 font-mono text-[11px] text-slate-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-slate-300 font-bold">{activeModel}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
