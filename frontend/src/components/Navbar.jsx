import React from 'react';
import { Activity, Brain, Database, Cpu, Sparkles, BarChart3 } from 'lucide-react';

export default function Navbar({ systemStatus, onOpenAnalytics }) {
  const isOnline = systemStatus?.status === 'online';
  const activeModel = systemStatus?.active_model || 'llama3.2:latest';
  const chunks = systemStatus?.chroma_chunks || 20;

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50 px-6 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-blue-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-tight text-white">NeuroMultimodal AI</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 font-medium">
                CNN • GNN • RAG
              </span>
            </div>
            <p className="text-xs text-slate-400">Clinical Neuro-Oncology Diagnostic & Decision Support Suite</p>
          </div>
        </div>

        {/* Status Indicators & Action */}
        <div className="flex items-center space-x-4 text-xs">
          {/* Ollama Status */}
          <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-slate-400">Ollama:</span>
            <span className="text-slate-200 font-mono font-medium">{activeModel}</span>
          </div>

          {/* Model Architecture */}
          <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">Backend:</span>
            <span className="text-slate-200 font-medium">ResNet18 + GAT + Fusion</span>
          </div>

          {/* Vector Store */}
          <div className="hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60">
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-400">RAG:</span>
            <span className="text-slate-200 font-medium">{chunks} WHO Guidelines</span>
          </div>

          {/* Analytics Button */}
          <button
            onClick={onOpenAnalytics}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/40 text-indigo-300 font-medium transition cursor-pointer"
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Ablation & Analytics</span>
          </button>
        </div>
      </div>
    </header>
  );
}
