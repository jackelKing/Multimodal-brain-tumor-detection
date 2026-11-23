import React from 'react';
import {
  BookOpen,
  Dna,
  Activity,
  Ruler,
  FileCheck,
  ShieldAlert,
  Sparkles,
  Layers,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';

export default function ClinicalGuideView() {
  return (
    <div className="space-y-6 animate-fadeIn text-slate-100">
      {/* Page Header */}
      <div className="p-5 sm:p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-xl">
        <div className="flex items-center space-x-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg sm:text-xl font-extrabold text-white tracking-tight">
              WHO CNS5 Clinical Reference &amp; Protocol Guide
            </h1>
            <p className="text-xs text-slate-400">
              Diagnostic guidelines, molecular classification criteria, RTOG RPA scoring, and treatment pathways.
            </p>
          </div>
        </div>
      </div>

      {/* Section 1: Molecular Biomarkers */}
      <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center space-x-2 text-indigo-400 font-bold uppercase tracking-wider text-xs">
          <Dna className="w-4 h-4" />
          <span>1. Essential WHO CNS5 Molecular Biomarkers</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-white text-sm">IDH1 / IDH2 Status</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-800 font-bold">
                Diagnostic
              </span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              <strong>Wildtype:</strong> In adults with diffuse astrocytic glioma, IDH-wildtype with microvascular proliferation, necrosis, or TERT promoter mutation defines <em>Glioblastoma, IDH-wildtype, WHO Grade 4</em>. Aggressive trajectory (median OS ~15 months).
            </p>
            <p className="text-xs text-slate-400 leading-relaxed pt-1 border-t border-slate-900">
              <strong>Mutant:</strong> Classified as Astrocytoma, IDH-mutant (Grades 2–4) or Oligodendroglioma. Significantly favorable survival (median OS 4–12+ years).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-white text-sm">MGMT Promoter</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-950 text-amber-300 border border-amber-800 font-bold">
                Predictive
              </span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              <strong>Hypermethylated:</strong> Epigenetic silencing of the $O^6$-methylguanine-DNA methyltransferase repair gene prevents tumor cells from repairing alkylating chemotherapy damage. Confers marked response and survival benefit with Temozolomide.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed pt-1 border-t border-slate-900">
              <strong>Unmethylated:</strong> Intact enzyme repairs DNA lesions; relative resistance to alkylators.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-white text-sm">1p/19q Co-deletion</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold">
                Definitive
              </span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              <strong>Co-deleted:</strong> Complete, unbalanced whole-arm loss of 1p and 19q is the defining genetic criterion for <em>Oligodendroglioma, IDH-mutant and 1p/19q-codeleted</em>. Chemosensitive to PCV and Temozolomide.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed pt-1 border-t border-slate-900">
              <strong>Intact (Non-codeleted):</strong> IDH-mutant tumors without 1p/19q loss are diagnosed as Astrocytoma.
            </p>
          </div>
        </div>
      </div>

      {/* Section 2: RTOG RPA Scoring Table */}
      <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center space-x-2 text-cyan-400 font-bold uppercase tracking-wider text-xs">
          <Activity className="w-4 h-4" />
          <span>2. RTOG Recursive Partitioning Analysis (RPA) for Glioblastoma</span>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="bg-slate-950 text-[11px] uppercase font-bold text-slate-400">
              <tr>
                <th className="p-3 border-b border-slate-800">RPA Class</th>
                <th className="p-3 border-b border-slate-800">Age</th>
                <th className="p-3 border-b border-slate-800">KPS Score</th>
                <th className="p-3 border-b border-slate-800">Surgical Extent</th>
                <th className="p-3 border-b border-slate-800">Neurological Function</th>
                <th className="p-3 border-b border-slate-800">Median OS</th>
                <th className="p-3 border-b border-slate-800">2-Year OS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
              <tr className="hover:bg-slate-950/40">
                <td className="p-3 font-bold text-emerald-400">Class III</td>
                <td className="p-3">&lt; 50 yrs</td>
                <td className="p-3">&ge; 90</td>
                <td className="p-3">Any</td>
                <td className="p-3">Normal / Able to Work</td>
                <td className="p-3 font-bold text-white">17.9 months</td>
                <td className="p-3 text-emerald-300">35%</td>
              </tr>
              <tr className="hover:bg-slate-950/40">
                <td className="p-3 font-bold text-cyan-400">Class IV</td>
                <td className="p-3">&ge; 50 yrs (or &lt; 50, KPS &lt; 90)</td>
                <td className="p-3">&ge; 70</td>
                <td className="p-3">GTR or STR</td>
                <td className="p-3">Preserved</td>
                <td className="p-3 font-bold text-white">14.5 months</td>
                <td className="p-3 text-cyan-300">18%</td>
              </tr>
              <tr className="hover:bg-slate-950/40">
                <td className="p-3 font-bold text-amber-400">Class V</td>
                <td className="p-3">&ge; 50 yrs</td>
                <td className="p-3">&ge; 70 (Biopsy) OR &lt; 70</td>
                <td className="p-3">Biopsy only / Any</td>
                <td className="p-3">Preserved / Mild</td>
                <td className="p-3 font-bold text-white">10.0 months</td>
                <td className="p-3 text-amber-300">8%</td>
              </tr>
              <tr className="hover:bg-slate-950/40">
                <td className="p-3 font-bold text-red-400">Class VI</td>
                <td className="p-3">&ge; 50 yrs</td>
                <td className="p-3">&lt; 70</td>
                <td className="p-3">Any</td>
                <td className="p-3">Neurological Deficit Present</td>
                <td className="p-3 font-bold text-white">4.6 months</td>
                <td className="p-3 text-red-400">2%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 3: Calipers & Stupp Protocols */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Calipers */}
        <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3">
          <div className="flex items-center space-x-2 text-emerald-400 font-bold uppercase tracking-wider text-xs">
            <Ruler className="w-4 h-4" />
            <span>3. Digital Caliper &amp; Volumetric Standards</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            The platform is calibrated to standardized DICOM pixel spacing ($0.9375\text{ mm/pixel}$).
            Two-point cross-sectional calipers measure the maximal orthogonal lesion dimensions ($d_1, d_2$).
          </p>
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-cyan-300 space-y-1">
            <div>Distance (mm) = &radic;[(x&sub2; - x&sub1;)&sup2; + (y&sub2; - y&sub1;)&sup2;] &times; 0.9375</div>
            <div>3D Ellipsoid Vol (cm&sup3;) = (&pi; / 6) &times; d&sub1; &times; d&sub2; &times; d&sub3; / 1000</div>
          </div>
          <p className="text-[11px] text-slate-400">
            Automated morphology segments <strong>Enhancing Tumor Core (ET)</strong>, <strong>Peritumoral Edema (ED)</strong>, and <strong>Central Necrosis (NCR)</strong> in line with BraTS consensus.
          </p>
        </div>

        {/* Stupp Protocol */}
        <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3">
          <div className="flex items-center space-x-2 text-purple-400 font-bold uppercase tracking-wider text-xs">
            <Sparkles className="w-4 h-4" />
            <span>4. Stupp Chemoradiotherapy Protocol (NCCN Standard)</span>
          </div>
          <div className="space-y-2 text-xs">
            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="font-bold text-white block">Concomitant Phase:</span>
              <span className="text-slate-300">
                Focal external beam radiation (60 Gy in 30 daily fractions of 2 Gy) + Temozolomide 75 mg/m&sup2;/day oral continuous for 6 weeks.
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="font-bold text-white block">Maintenance Phase:</span>
              <span className="text-slate-300">
                4-week break, followed by 6 cycles of adjuvant Temozolomide: 150–200 mg/m&sup2;/day on Days 1–5 every 28 days.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
