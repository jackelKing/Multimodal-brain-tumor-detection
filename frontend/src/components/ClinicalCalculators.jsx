import React, { useState } from 'react';
import { Calculator, Dna, Activity, TrendingUp, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ClinicalCalculators({ patientInfo, results }) {
  // RPA inputs
  const [age, setAge] = useState(patientInfo?.age || 58);
  const [kps, setKps] = useState(patientInfo?.kps || 80);
  const [resectionStatus, setResectionStatus] = useState('GTR');
  const [neuroFunction, setNeuroFunction] = useState('Normal');

  // Compute RPA Class dynamically
  const calculateRpa = () => {
    const isResection = resectionStatus === 'GTR' || resectionStatus === 'STR';
    const isNormalNeuro = neuroFunction === 'Normal';

    if (age < 50) {
      if (kps >= 90) {
        return {
          rpaClass: 'Class III',
          medianOs: '17.9 months',
          twoYearOs: '35%',
          riskTier: 'Favorable (Grade IV GBM)',
          color: 'text-emerald-400 bg-emerald-950/40 border-emerald-800'
        };
      } else {
        return {
          rpaClass: 'Class IV',
          medianOs: '14.8 months',
          twoYearOs: '20%',
          riskTier: 'Intermediate Risk',
          color: 'text-amber-400 bg-amber-950/40 border-amber-800'
        };
      }
    } else {
      if (kps >= 70) {
        if (isResection) {
          return {
            rpaClass: 'Class IV',
            medianOs: '14.5 months',
            twoYearOs: '18%',
            riskTier: 'Intermediate Risk',
            color: 'text-amber-400 bg-amber-950/40 border-amber-800'
          };
        } else {
          return {
            rpaClass: 'Class V',
            medianOs: '10.0 months',
            twoYearOs: '8%',
            riskTier: 'High Risk',
            color: 'text-rose-400 bg-rose-950/40 border-rose-800'
          };
        }
      } else {
        if (isNormalNeuro) {
          return {
            rpaClass: 'Class V',
            medianOs: '8.9 months',
            twoYearOs: '6%',
            riskTier: 'High Risk',
            color: 'text-rose-400 bg-rose-950/40 border-rose-800'
          };
        } else {
          return {
            rpaClass: 'Class VI',
            medianOs: '4.6 months',
            twoYearOs: '2%',
            riskTier: 'Very High Risk / Frail',
            color: 'text-red-500 bg-red-950/40 border-red-800'
          };
        }
      }
    }
  };

  const rpa = calculateRpa();

  // ECOG performance score derived from KPS
  const getEcogScore = (val) => {
    if (val >= 90) return { ecog: 0, desc: 'Fully active, able to carry on all pre-disease activities' };
    if (val >= 70) return { ecog: 1, desc: 'Restricted in strenuous activity, ambulatory and capable of light work' };
    if (val >= 50) return { ecog: 2, desc: 'Ambulatory and capable of all self-care, up and about > 50% of waking hours' };
    if (val >= 30) return { ecog: 3, desc: 'Capable of only limited self-care, confined to bed/chair > 50% of waking hours' };
    return { ecog: 4, desc: 'Completely disabled, cannot carry on any self-care' };
  };

  const ecog = getEcogScore(kps);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white flex items-center space-x-2">
          <Calculator className="w-5 h-5 text-indigo-400" />
          <span>Neuro-Oncology Clinical Prognostic Calculators</span>
        </h2>
        <p className="text-xs text-slate-400">
          RTOG Recursive Partitioning Analysis (RPA) &amp; ECOG Performance Stratification.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
        {/* Left: RTOG RPA Calculator */}
        <div className="space-y-4 p-4 rounded-xl bg-slate-950 border border-slate-800">
          <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
            <TrendingUp className="w-4 h-4 text-cyan-400" />
            <span>RTOG RPA Glioblastoma Stratification</span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="text-slate-400 block mb-1">Age</label>
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(Number(e.target.value))}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-white"
              />
            </div>
            <div>
              <label className="text-slate-400 block mb-1">KPS Score ({kps})</label>
              <input
                type="range"
                min="30"
                max="100"
                step="5"
                value={kps}
                onChange={(e) => setKps(Number(e.target.value))}
                className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded cursor-pointer mt-2"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="text-slate-400 block mb-1">Surgical Resection Extent</label>
              <select
                value={resectionStatus}
                onChange={(e) => setResectionStatus(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1.5 text-white"
              >
                <option value="GTR">Gross Total Resection (GTR)</option>
                <option value="STR">Subtotal Resection (STR)</option>
                <option value="Biopsy">Biopsy Only</option>
              </select>
            </div>
            <div>
              <label className="text-slate-400 block mb-1">Neurological Status</label>
              <select
                value={neuroFunction}
                onChange={(e) => setNeuroFunction(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1.5 text-white"
              >
                <option value="Normal">Able to Work / Preserved</option>
                <option value="Deficit">Neurological Deficit Present</option>
              </select>
            </div>
          </div>

          {/* RPA Outcome Banner */}
          <div className={`p-3.5 rounded-xl border ${rpa.color} space-y-1.5`}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider">RTOG {rpa.rpaClass}</span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-900/60 border border-current">
                {rpa.riskTier}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs pt-1">
              <div>
                <span className="text-slate-400 block text-[10px]">Median Overall Survival:</span>
                <span className="font-extrabold text-white text-sm">{rpa.medianOs}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px]">2-Year Survival Rate:</span>
                <span className="font-extrabold text-white text-sm">{rpa.twoYearOs}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Functional Performance & Molecular Risk */}
        <div className="space-y-4 p-4 rounded-xl bg-slate-950 border border-slate-800">
          <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
            <Activity className="w-4 h-4 text-emerald-400" />
            <span>Performance Status Converter</span>
          </div>

          {/* KPS to ECOG Converter */}
          <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5 text-xs">
            <div className="flex justify-between items-center">
              <span className="font-semibold text-slate-300">KPS: {kps} / 100</span>
              <span className="font-bold text-indigo-400 font-mono px-2 py-0.5 rounded bg-indigo-950/60 border border-indigo-800">
                ECOG Score: {ecog.ecog}
              </span>
            </div>
            <p className="text-[11px] text-slate-400">{ecog.desc}</p>
          </div>

          {/* Molecular Risk Profile */}
          <div className="space-y-2 text-xs pt-2">
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center space-x-1">
              <Dna className="w-3.5 h-3.5 text-cyan-400" />
              <span>WHO CNS5 Molecular Subtype Prognostic Tier</span>
            </div>

            <div className="space-y-1.5 text-[11px]">
              <div className="p-2 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-emerald-300 flex justify-between items-center">
                <span>Oligodendroglioma (IDH-mutant, 1p/19q codeleted)</span>
                <span className="font-bold font-mono">OS: 10 - 15+ yrs</span>
              </div>
              <div className="p-2 rounded-lg bg-cyan-950/30 border border-cyan-800/40 text-cyan-300 flex justify-between items-center">
                <span>Astrocytoma (IDH-mutant, non-codeleted)</span>
                <span className="font-bold font-mono">OS: 4 - 8 yrs</span>
              </div>
              <div className="p-2 rounded-lg bg-red-950/30 border border-red-800/40 text-red-300 flex justify-between items-center">
                <span>Glioblastoma (IDH-wildtype, WHO Grade 4)</span>
                <span className="font-bold font-mono">OS: 14 - 18 mos</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
