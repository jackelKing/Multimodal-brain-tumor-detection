import React from 'react';
import { ShieldAlert, CheckCircle, AlertTriangle, Activity, PieChart, Sparkles } from 'lucide-react';

export default function DiagnosticResults({ results }) {
  if (!results) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center text-slate-500 space-y-3">
        <Activity className="w-10 h-10 mx-auto opacity-30 animate-pulse text-indigo-400" />
        <h3 className="text-sm font-semibold text-slate-300">Awaiting Multimodal Analysis</h3>
        <p className="text-xs max-w-sm mx-auto">
          Click "Run Multimodal AI Analysis" to fuse CNN spatial features, GNN topological graph reasoning, and clinical biomarkers.
        </p>
      </div>
    );
  }

  const {
    predicted_class,
    confidence,
    class_probabilities,
    is_tumor_detected,
    detection_confidence,
    predicted_grade,
    grade_confidence,
    gate_weights
  } = results;

  // Calculate normalized gating weights
  const cnnW = gate_weights?.cnn_weight || 0.33;
  const gnnW = gate_weights?.gnn_weight || 0.33;
  const clinW = gate_weights?.clinical_weight || 0.33;
  const totW = cnnW + gnnW + clinW || 1.0;
  const cnnPct = Math.round((cnnW / totW) * 100);
  const gnnPct = Math.round((gnnW / totW) * 100);
  const clinPct = Math.round((clinW / totW) * 100);

  // Badge style for predicted class
  let classColor = 'from-red-600 to-rose-700 border-red-500/40 text-red-200';
  if (predicted_class.includes('Pituitary')) classColor = 'from-blue-600 to-indigo-700 border-blue-500/40 text-blue-200';
  else if (predicted_class.includes('Meningioma')) classColor = 'from-amber-600 to-yellow-700 border-amber-500/40 text-amber-200';
  else if (predicted_class.includes('Normal') || predicted_class.includes('No Tumor')) classColor = 'from-emerald-600 to-green-700 border-emerald-500/40 text-emerald-200';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Card 1: Primary Diagnosis */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 relative overflow-hidden">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">
            Primary Pathology
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-xl font-extrabold text-white">{predicted_class}</span>
            <span className="text-xs font-mono font-bold text-indigo-400">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="mt-2.5">
            <span className={`inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full border bg-gradient-to-r ${classColor}`}>
              High Confidence
            </span>
          </div>
        </div>

        {/* Card 2: Binary Detection */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 relative overflow-hidden">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">
            Detection Status
          </div>
          <div className="flex items-center space-x-2">
            {is_tumor_detected ? (
              <ShieldAlert className="w-5 h-5 text-red-400" />
            ) : (
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            )}
            <span className={`text-base font-bold ${is_tumor_detected ? 'text-red-400' : 'text-emerald-400'}`}>
              {is_tumor_detected ? 'Tumor Detected' : 'Normal / No Tumor'}
            </span>
          </div>
          <div className="text-xs font-mono text-slate-400 mt-2">
            Confidence: <span className="text-slate-200 font-bold">{(detection_confidence * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* Card 3: WHO Histological Grade */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 relative overflow-hidden">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">
            Prognostic WHO Grade
          </div>
          <div className="text-base font-bold text-purple-300">
            {predicted_grade}
          </div>
          <div className="text-xs font-mono text-slate-400 mt-2">
            Risk Confidence: <span className="text-purple-400 font-bold">{(grade_confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Probability Bars */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Subtype Probability Distribution
          </span>
          <span className="text-[11px] text-slate-500">Multiclass Softmax</span>
        </div>

        <div className="space-y-2.5">
          {Object.entries(class_probabilities || {}).map(([cls, prob]) => {
            const pct = (prob * 100).toFixed(1);
            let barColor = 'bg-slate-700';
            if (cls.includes('Glioma')) barColor = 'bg-red-500';
            else if (cls.includes('Meningioma')) barColor = 'bg-amber-500';
            else if (cls.includes('Pituitary')) barColor = 'bg-blue-500';
            else if (cls.includes('Normal') || cls.includes('No Tumor')) barColor = 'bg-emerald-500';

            return (
              <div key={cls} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">{cls}</span>
                  <span className="font-mono text-slate-400 font-semibold">{pct}%</span>
                </div>
                <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className={`h-full ${barColor} rounded-full transition-all duration-700`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Multimodal Gating Breakdown */}
      <div className="border-t border-slate-800/80 pt-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
            <PieChart className="w-3.5 h-3.5 text-cyan-400" />
            <span>Multimodal Gating Weights</span>
          </span>
          <span className="text-[11px] text-slate-400">Dynamic Decision Contribution</span>
        </div>

        <div className="grid grid-cols-3 gap-2.5 text-center">
          <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800/80">
            <div className="text-[10px] text-indigo-400 font-bold uppercase">CNN Visual</div>
            <div className="text-base font-extrabold text-white mt-0.5">{cnnPct}%</div>
            <div className="w-full bg-slate-800 h-1 rounded-full mt-1.5 overflow-hidden">
              <div className="bg-indigo-500 h-full" style={{ width: `${cnnPct}%` }} />
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800/80">
            <div className="text-[10px] text-cyan-400 font-bold uppercase">GNN Topology</div>
            <div className="text-base font-extrabold text-white mt-0.5">{gnnPct}%</div>
            <div className="w-full bg-slate-800 h-1 rounded-full mt-1.5 overflow-hidden">
              <div className="bg-cyan-500 h-full" style={{ width: `${gnnPct}%` }} />
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800/80">
            <div className="text-[10px] text-purple-400 font-bold uppercase">Clinical Prior</div>
            <div className="text-base font-extrabold text-white mt-0.5">{clinPct}%</div>
            <div className="w-full bg-slate-800 h-1 rounded-full mt-1.5 overflow-hidden">
              <div className="bg-purple-500 h-full" style={{ width: `${clinPct}%` }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
