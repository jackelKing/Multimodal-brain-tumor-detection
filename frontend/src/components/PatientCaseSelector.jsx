import React, { useState, useRef } from 'react';
import {
  Upload,
  User,
  Dna,
  Activity,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Play,
  RotateCcw
} from 'lucide-react';

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
  const [showBiomarkerEditor, setShowBiomarkerEditor] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onCustomImageUpload(file);
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800/80 rounded-2xl p-4 sm:p-5 shadow-xl space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center">
            <User className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white tracking-tight">Patient Case &amp; Scan</h2>
            <p className="text-[11px] text-slate-400">Select benchmark cohort or upload scan</p>
          </div>
        </div>

        {/* Custom Upload Trigger */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[11px] text-slate-300 transition cursor-pointer font-medium"
        >
          <Upload className="w-3.5 h-3.5 text-cyan-400" />
          <span>Upload MRI</span>
        </button>
      </div>

      {/* Preset Cohort Selector - Clean Horizontal/Grid Pills */}
      <div className="grid grid-cols-2 gap-2">
        {cases.map((c) => {
          const isSelected = selectedCaseId === c.id;
          let badgeBorder = 'border-slate-800 text-slate-400';
          if (c.target_class.includes('Glioma')) badgeBorder = isSelected ? 'border-red-500 bg-red-950/40 text-red-200' : 'border-slate-800 text-slate-300';
          else if (c.target_class.includes('Meningioma')) badgeBorder = isSelected ? 'border-amber-500 bg-amber-950/40 text-amber-200' : 'border-slate-800 text-slate-300';
          else if (c.target_class.includes('Pituitary')) badgeBorder = isSelected ? 'border-blue-500 bg-blue-950/40 text-blue-200' : 'border-slate-800 text-slate-300';
          else if (c.target_class.includes('Normal') || c.target_class.includes('No Tumor')) badgeBorder = isSelected ? 'border-emerald-500 bg-emerald-950/40 text-emerald-200' : 'border-slate-800 text-slate-300';

          return (
            <button
              key={c.id}
              onClick={() => onSelectCase(c)}
              className={`p-2.5 rounded-xl border text-left transition-all cursor-pointer relative flex flex-col justify-between ${
                isSelected
                  ? `${badgeBorder} shadow-lg shadow-indigo-500/10 bg-indigo-950/30`
                  : 'border-slate-800/80 bg-slate-950/60 hover:bg-slate-800/40 text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className="flex items-center justify-between w-full mb-1">
                <span className="text-[10px] font-bold uppercase tracking-wider truncate">
                  {c.target_class}
                </span>
                {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400 shrink-0 ml-1" />}
              </div>
              <div className="text-xs font-semibold text-white truncate w-full">{c.title}</div>
            </button>
          );
        })}
      </div>

      {/* Compact Patient Profile Card */}
      <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center space-x-1.5 font-bold text-slate-300">
            <Dna className="w-3.5 h-3.5 text-cyan-400" />
            <span>Clinical Demographics &amp; Biomarkers</span>
          </div>
          <button
            onClick={() => setShowBiomarkerEditor(!showBiomarkerEditor)}
            className="text-[11px] text-cyan-400 hover:text-cyan-300 font-medium flex items-center space-x-1 transition cursor-pointer"
          >
            <span>{showBiomarkerEditor ? 'Done' : 'Edit'}</span>
            {showBiomarkerEditor ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        </div>

        {/* Read-Only Badge Grid (when not editing) */}
        {!showBiomarkerEditor ? (
          <div className="flex flex-wrap gap-1.5 text-[11px] pt-1">
            <span className="px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
              Age: <strong className="text-white">{patientInfo.age}y</strong>
            </span>
            <span className="px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
              Gender: <strong className="text-white">{patientInfo.gender}</strong>
            </span>
            <span className="px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
              KPS: <strong className="text-indigo-300">{patientInfo.kps}/100</strong>
            </span>
            <span className={`px-2 py-0.5 rounded-md border ${
              patientInfo.idh1 === 'Mutant'
                ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                : 'bg-red-950/60 border-red-800 text-red-300'
            }`}>
              IDH1: <strong>{patientInfo.idh1}</strong>
            </span>
            <span className={`px-2 py-0.5 rounded-md border ${
              patientInfo.mgmt === 'Methylated'
                ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                : 'bg-slate-900 border-slate-800 text-slate-300'
            }`}>
              MGMT: <strong>{patientInfo.mgmt}</strong>
            </span>
            <span className="px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
              1p/19q: <strong>{patientInfo.codeletion_1p19q}</strong>
            </span>
          </div>
        ) : (
          /* Expandable Form (when editing) */
          <div className="space-y-3 pt-2 text-xs border-t border-slate-800">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-slate-400 block mb-0.5 text-[10px]">Age</label>
                <input
                  type="number"
                  value={patientInfo.age}
                  onChange={(e) => onUpdatePatientInfo({ ...patientInfo, age: Number(e.target.value) })}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1 text-white text-xs"
                />
              </div>
              <div>
                <label className="text-slate-400 block mb-0.5 text-[10px]">Gender</label>
                <select
                  value={patientInfo.gender}
                  onChange={(e) => onUpdatePatientInfo({ ...patientInfo, gender: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1 text-white text-xs"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-[10px] text-slate-400 mb-0.5">
                <span>Karnofsky Performance Scale (KPS)</span>
                <span className="font-mono text-indigo-400 font-bold">{patientInfo.kps} / 100</span>
              </div>
              <input
                type="range"
                min="30"
                max="100"
                step="5"
                value={patientInfo.kps}
                onChange={(e) => onUpdatePatientInfo({ ...patientInfo, kps: Number(e.target.value) })}
                className="w-full accent-indigo-500 h-1 bg-slate-800 rounded cursor-pointer"
              />
            </div>

            <div className="grid grid-cols-3 gap-2 text-[11px]">
              <div>
                <label className="text-slate-400 block mb-0.5 text-[10px]">IDH1 Status</label>
                <select
                  value={patientInfo.idh1}
                  onChange={(e) => onUpdatePatientInfo({ ...patientInfo, idh1: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-1.5 py-1 text-white text-[11px]"
                >
                  <option value="Wildtype">Wildtype</option>
                  <option value="Mutant">Mutant</option>
                </select>
              </div>
              <div>
                <label className="text-slate-400 block mb-0.5 text-[10px]">MGMT Promoter</label>
                <select
                  value={patientInfo.mgmt}
                  onChange={(e) => onUpdatePatientInfo({ ...patientInfo, mgmt: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-1.5 py-1 text-white text-[11px]"
                >
                  <option value="Unmethylated">Unmethylated</option>
                  <option value="Methylated">Methylated</option>
                </select>
              </div>
              <div>
                <label className="text-slate-400 block mb-0.5 text-[10px]">1p/19q Status</label>
                <select
                  value={patientInfo.codeletion_1p19q}
                  onChange={(e) => onUpdatePatientInfo({ ...patientInfo, codeletion_1p19q: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-1.5 py-1 text-white text-[11px]"
                >
                  <option value="Non-codeleted">Non-codeleted</option>
                  <option value="Codeleted">Codeleted</option>
                </select>
              </div>
            </div>

            <div>
              <label className="text-slate-400 block mb-0.5 text-[10px]">Presenting Symptoms</label>
              <input
                type="text"
                value={patientInfo.symptoms}
                onChange={(e) => onUpdatePatientInfo({ ...patientInfo, symptoms: e.target.value })}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1 text-white text-xs"
              />
            </div>
          </div>
        )}
      </div>

      {/* Primary Action Button */}
      <button
        onClick={onRunAnalysis}
        disabled={loading}
        className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/25 transition cursor-pointer disabled:opacity-50 flex items-center justify-center space-x-2"
      >
        {loading ? (
          <>
            <Sparkles className="w-4 h-4 animate-spin text-white" />
            <span>Fusing CNN + GATv2 + Biomarkers...</span>
          </>
        ) : (
          <>
            <Play className="w-4 h-4 fill-current text-white" />
            <span>Run Multimodal AI Analysis</span>
          </>
        )}
      </button>
    </div>
  );
}
