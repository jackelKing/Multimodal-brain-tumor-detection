import React, { useRef } from 'react';
import { Upload, FileText, User, Dna, Activity, CheckCircle2, ChevronRight, Sparkles } from 'lucide-react';

export default function PatientCaseSelector({
  cases,
  selectedCaseId,
  onSelectCase,
  patientInfo,
  onUpdatePatientInfo,
  onCustomImageUpload,
  onRunAnalysis,
  loading
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onCustomImageUpload(file);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-bold text-white flex items-center space-x-2">
          <User className="w-5 h-5 text-indigo-400" />
          <span>Patient & Neuroimaging Input</span>
        </h2>
        <p className="text-xs text-slate-400">Select a verified clinical benchmark case or upload a custom MRI scan.</p>
      </div>

      {/* Preset Cases Selector */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center justify-between">
          <span>Clinical Case Library</span>
          <span className="text-indigo-400 font-normal">4 Categories</span>
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          {cases.map((c) => {
            const isSelected = selectedCaseId === c.id;
            let badgeBg = 'bg-slate-800 text-slate-300 border-slate-700';
            if (c.target_class.includes('Glioma')) badgeBg = 'bg-red-950/60 text-red-300 border-red-800/60';
            else if (c.target_class.includes('Pituitary')) badgeBg = 'bg-blue-950/60 text-blue-300 border-blue-800/60';
            else if (c.target_class.includes('Meningioma')) badgeBg = 'bg-amber-950/60 text-amber-300 border-amber-800/60';
            else if (c.target_class.includes('Normal') || c.target_class.includes('No Tumor')) badgeBg = 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60';

            return (
              <button
                key={c.id}
                onClick={() => onSelectCase(c)}
                className={`p-3 rounded-xl border text-left transition-all cursor-pointer relative overflow-hidden flex flex-col justify-between ${
                  isSelected
                    ? 'border-indigo-500 bg-indigo-950/30 shadow-lg shadow-indigo-500/10'
                    : 'border-slate-800 bg-slate-950/50 hover:border-slate-700 hover:bg-slate-800/50'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full border ${badgeBg}`}>
                      {c.target_class}
                    </span>
                    {isSelected && <CheckCircle2 className="w-4 h-4 text-indigo-400" />}
                  </div>
                  <h4 className="text-xs font-semibold text-white leading-snug">{c.title}</h4>
                </div>
                <p className="text-[11px] text-slate-400 mt-2 line-clamp-2">{c.description}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Custom Upload Button */}
      <div className="pt-1">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="w-full py-2.5 px-4 rounded-xl border border-dashed border-slate-700 hover:border-indigo-500 bg-slate-950/40 hover:bg-indigo-950/20 text-xs font-medium text-slate-300 hover:text-indigo-300 flex items-center justify-center space-x-2 transition cursor-pointer"
        >
          <Upload className="w-4 h-4 text-slate-400" />
          <span>Upload Custom MRI Scan (DICOM / JPEG / PNG)</span>
        </button>
      </div>

      {/* Clinical Profile Editor */}
      <div className="border-t border-slate-800/80 pt-4 space-y-4">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
            <Dna className="w-3.5 h-3.5 text-cyan-400" />
            <span>Clinical Demographics & Biomarkers</span>
          </label>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-[11px] text-slate-400 block mb-1">Patient Age</label>
            <input
              type="number"
              value={patientInfo.age || 52}
              onChange={(e) => onUpdatePatientInfo({ ...patientInfo, age: parseInt(e.target.value) || 50 })}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
            />
          </div>
          <div>
            <label className="text-[11px] text-slate-400 block mb-1">Gender</label>
            <select
              value={patientInfo.gender || 'Male'}
              onChange={(e) => onUpdatePatientInfo({ ...patientInfo, gender: e.target.value })}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <label className="text-[11px] text-slate-400">Karnofsky Performance Score (KPS)</label>
            <span className="text-xs font-mono font-bold text-indigo-400">{patientInfo.kps || 80} / 100</span>
          </div>
          <input
            type="range"
            min="40"
            max="100"
            step="5"
            value={patientInfo.kps || 80}
            onChange={(e) => onUpdatePatientInfo({ ...patientInfo, kps: parseInt(e.target.value) })}
            className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-[11px] text-slate-400 block mb-1">IDH1 Status</label>
            <select
              value={patientInfo.idh1 || 'Wildtype'}
              onChange={(e) => onUpdatePatientInfo({ ...patientInfo, idh1: e.target.value })}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
            >
              <option value="Wildtype">Wildtype</option>
              <option value="Mutant (R132H)">Mutant (R132H)</option>
              <option value="Normal">Normal</option>
            </select>
          </div>
          <div>
            <label className="text-[11px] text-slate-400 block mb-1">MGMT Promoter</label>
            <select
              value={patientInfo.mgmt || 'Unmethylated'}
              onChange={(e) => onUpdatePatientInfo({ ...patientInfo, mgmt: e.target.value })}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
            >
              <option value="Unmethylated">Unmethylated</option>
              <option value="Methylated">Methylated</option>
              <option value="Normal">Normal</option>
            </select>
          </div>
        </div>

        <div>
          <label className="text-[11px] text-slate-400 block mb-1">Chief Neurological Symptoms</label>
          <textarea
            rows="2"
            value={patientInfo.symptoms || ''}
            onChange={(e) => onUpdatePatientInfo({ ...patientInfo, symptoms: e.target.value })}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-indigo-500"
            placeholder="e.g. Progressive headaches, focal seizures, hemiparesis..."
          />
        </div>
      </div>

      {/* Action Button */}
      <button
        onClick={onRunAnalysis}
        disabled={loading}
        className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-500 hover:from-indigo-500 hover:to-cyan-400 text-white font-semibold text-sm shadow-lg shadow-indigo-500/25 transition flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-50"
      >
        {loading ? (
          <>
            <Activity className="w-4 h-4 animate-spin text-white" />
            <span>Fusing CNN, GNN & Biomarkers...</span>
          </>
        ) : (
          <>
            <Sparkles className="w-4 h-4 text-white" />
            <span>Run Multimodal AI Analysis</span>
            <ChevronRight className="w-4 h-4 text-white/80" />
          </>
        )}
      </button>
    </div>
  );
}
