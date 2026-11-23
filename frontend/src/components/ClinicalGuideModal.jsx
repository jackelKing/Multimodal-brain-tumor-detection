import React from 'react';
import {
  BookOpen,
  X,
  Dna,
  Activity,
  Layers,
  Ruler,
  AlertTriangle,
  FileCheck,
  CheckCircle2
} from 'lucide-react';

export default function ClinicalGuideModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="relative w-full max-w-4xl bg-slate-900 border border-slate-700/80 rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-950">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">WHO CNS5 Clinical Reference &amp; Protocol Guide</h2>
              <p className="text-xs text-slate-400">Diagnostic criteria, molecular classification, and prognostic scoring.</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 overflow-y-auto custom-scrollbar flex-1 text-xs text-slate-300">
          {/* Section 1: Molecular Biomarkers */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2 text-indigo-400 font-bold uppercase tracking-wider text-[11px]">
              <Dna className="w-4 h-4" />
              <span>1. Essential WHO CNS5 Molecular Biomarkers</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
                <div className="font-bold text-white flex items-center justify-between">
                  <span>IDH1 / IDH2 Status</span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">Diagnostic</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  <strong>Wildtype:</strong> Defines primary Glioblastoma WHO Grade 4. Aggressive course with median OS ~15 months.<br/>
                  <strong>Mutant:</strong> Astrocytoma (Grade 2–4) or Oligodendroglioma. Significantly favorable prognosis (median OS 4–12+ yrs).
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
                <div className="font-bold text-white flex items-center justify-between">
                  <span>MGMT Promoter</span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800">Predictive</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  <strong>Hypermethylated:</strong> Epigenetic silencing of DNA repair enzyme MGMT. Confers superior sensitivity to alkylating chemotherapy (Temozolomide).<br/>
                  <strong>Unmethylated:</strong> Relative resistance to alkylating therapy; consider radiotherapy dose optimization.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
                <div className="font-bold text-white flex items-center justify-between">
                  <span>1p/19q Co-deletion</span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">Definitive</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  <strong>Present (Co-deleted):</strong> Obligate molecular hallmark of Oligodendroglioma. Chemosensitive with median survival extending past 12–15 years.<br/>
                  <strong>Absent (Intact):</strong> IDH-mutant tumors are classified as Astrocytoma.
                </p>
              </div>
            </div>
          </div>

          {/* Section 2: RTOG RPA Scoring */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2 text-cyan-400 font-bold uppercase tracking-wider text-[11px]">
              <Activity className="w-4 h-4" />
              <span>2. RTOG Recursive Partitioning Analysis (RPA)</span>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left border-collapse">
                <thead className="bg-slate-950 text-[10px] uppercase font-bold text-slate-400">
                  <tr>
                    <th className="p-2.5 border-b border-slate-800">Class</th>
                    <th className="p-2.5 border-b border-slate-800">Age</th>
                    <th className="p-2.5 border-b border-slate-800">KPS Score</th>
                    <th className="p-2.5 border-b border-slate-800">Surgical Extent</th>
                    <th className="p-2.5 border-b border-slate-800">Median OS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                  <tr className="hover:bg-slate-950/40">
                    <td className="p-2.5 font-bold text-emerald-400">Class III</td>
                    <td className="p-2.5">&lt; 50 yrs</td>
                    <td className="p-2.5">&ge; 90</td>
                    <td className="p-2.5">Any</td>
                    <td className="p-2.5 font-bold text-white">17.9 months</td>
                  </tr>
                  <tr className="hover:bg-slate-950/40">
                    <td className="p-2.5 font-bold text-cyan-400">Class IV</td>
                    <td className="p-2.5">&ge; 50 yrs</td>
                    <td className="p-2.5">&ge; 70</td>
                    <td className="p-2.5">GTR or STR</td>
                    <td className="p-2.5 font-bold text-white">14.5 months</td>
                  </tr>
                  <tr className="hover:bg-slate-950/40">
                    <td className="p-2.5 font-bold text-amber-400">Class V</td>
                    <td className="p-2.5">&ge; 50 yrs</td>
                    <td className="p-2.5">&ge; 70 (Biopsy) OR &lt; 70 (Preserved Neuro)</td>
                    <td className="p-2.5">Biopsy / Any</td>
                    <td className="p-2.5 font-bold text-white">10.0 months</td>
                  </tr>
                  <tr className="hover:bg-slate-950/40">
                    <td className="p-2.5 font-bold text-red-400">Class VI</td>
                    <td className="p-2.5">&ge; 50 yrs</td>
                    <td className="p-2.5">&lt; 70</td>
                    <td className="p-2.5">Neurological Deficit</td>
                    <td className="p-2.5 font-bold text-white">4.6 months</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 3: Digital Caliper & Volumetric Principles */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2 text-emerald-400 font-bold uppercase tracking-wider text-[11px]">
              <Ruler className="w-4 h-4" />
              <span>3. Sub-millimeter Digital Caliper &amp; Ellipsoid Volumetrics</span>
            </div>
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <p className="text-[11px] text-slate-300 leading-relaxed">
                Neuroimaging dimensions are calibrated to standardized DICOM pixel spacing (0.9375 mm/pixel).
                When measuring cross-sectional diameter with the two-point caliper:
              </p>
              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 font-mono text-[11px] text-cyan-300">
                d = √[(x₂ - x₁)² + (y₂ - y₁)²] × 0.9375 mm<br/>
                Volume (cm³) = (π / 6) × d³ / 1000
              </div>
              <p className="text-[11px] text-slate-400">
                Compartmental masks delineate <strong>Enhancing Tumor Core (ET)</strong>, <strong>Peritumoral Edema (ED)</strong>, and <strong>Central Necrosis (NCR)</strong> in compliance with BraTS consensus standards.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition cursor-pointer"
          >
            Close Guide
          </button>
        </div>
      </div>
    </div>
  );
}
