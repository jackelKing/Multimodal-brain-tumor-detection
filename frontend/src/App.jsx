import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import PatientCaseSelector from './components/PatientCaseSelector';
import MriViewer from './components/MriViewer';
import DiagnosticResults from './components/DiagnosticResults';
import RagConsultation from './components/RagConsultation';
import ClinicalCalculators from './components/ClinicalCalculators';
import AnalyticsModal from './components/AnalyticsModal';
import { Eye, Calculator, Sparkles, LayoutDashboard } from 'lucide-react';

export default function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [cases, setCases] = useState([]);
  const [selectedCaseId, setSelectedCaseId] = useState(null);
  const [patientInfo, setPatientInfo] = useState({
    age: 58,
    gender: 'Male',
    symptoms: 'New-onset focal motor seizures, worsening headaches, left-sided hemiparesis',
    kps: 75,
    idh1: 'Wildtype',
    mgmt: 'Unmethylated',
    codeletion_1p19q: 'Non-codeleted'
  });

  const [activeWorkflowTab, setActiveWorkflowTab] = useState('all'); // 'all', 'imaging', 'calculators', 'rag'

  const [uploadedFile, setUploadedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [camOverlay, setCamOverlay] = useState(null);
  const [gnnOverlay, setGnnOverlay] = useState(null);
  const [segOverlay, setSegOverlay] = useState(null);

  const [results, setResults] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [selectedModel, setSelectedModel] = useState('llama3.2:latest');

  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [loadingReport, setLoadingReport] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [analyticsOpen, setAnalyticsOpen] = useState(false);
  const [metricsData, setMetricsData] = useState(null);

  // 1. Fetch initial status, cases, and metrics
  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => {
        setSystemStatus(data);
        if (data.active_model) setSelectedModel(data.active_model);
      })
      .catch((err) => console.error('Failed to fetch status:', err));

    fetch('/api/cases')
      .then((res) => res.json())
      .then((data) => {
        if (data.cases && data.cases.length > 0) {
          setCases(data.cases);
          const defaultCase = data.cases[0];
          setSelectedCaseId(defaultCase.id);
          setPatientInfo(defaultCase.patient_info);
          if (defaultCase.image_base64) {
            setOriginalImage(`data:image/jpeg;base64,${defaultCase.image_base64}`);
          }
        }
      })
      .catch((err) => console.error('Failed to fetch cases:', err));

    fetch('/api/metrics')
      .then((res) => res.json())
      .then((data) => setMetricsData(data))
      .catch((err) => console.error('Failed to fetch metrics:', err));
  }, []);

  // Handle preset case selection
  const handleSelectCase = (c) => {
    setSelectedCaseId(c.id);
    setPatientInfo(c.patient_info);
    setUploadedFile(null);
    if (c.image_base64) {
      setOriginalImage(`data:image/jpeg;base64,${c.image_base64}`);
    }
    setCamOverlay(null);
    setGnnOverlay(null);
    setSegOverlay(null);
    setResults(null);
    setReportData(null);
  };

  // Handle custom image upload
  const handleCustomImageUpload = (file) => {
    setUploadedFile(file);
    setSelectedCaseId(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage(e.target.result);
      setCamOverlay(null);
      setGnnOverlay(null);
      setSegOverlay(null);
      setResults(null);
      setReportData(null);
    };
    reader.readAsDataURL(file);
  };

  // Run Multimodal Inference
  const handleRunAnalysis = async () => {
    setLoadingAnalysis(true);
    setReportData(null);

    try {
      const formData = new FormData();
      if (uploadedFile) {
        formData.append('file', uploadedFile);
      } else if (selectedCaseId) {
        formData.append('case_id', selectedCaseId);
      } else if (originalImage) {
        formData.append('image_base64', originalImage);
      }

      formData.append('patient_info', JSON.stringify(patientInfo));

      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed');
      }

      const data = await res.json();
      setResults(data);

      if (data.cam_overlay_base64) {
        setCamOverlay(`data:image/jpeg;base64,${data.cam_overlay_base64}`);
      }
      if (data.gnn_overlay_base64) {
        setGnnOverlay(`data:image/jpeg;base64,${data.gnn_overlay_base64}`);
      }
      if (data.seg_overlay_base64) {
        setSegOverlay(`data:image/jpeg;base64,${data.seg_overlay_base64}`);
      }
      if (data.original_image_base64 && !originalImage) {
        setOriginalImage(`data:image/jpeg;base64,${data.original_image_base64}`);
      }
    } catch (err) {
      alert(`Inference Error: ${err.message}`);
    } finally {
      setLoadingAnalysis(false);
    }
  };

  // Generate RAG Clinical Report
  const handleGenerateReport = async () => {
    if (!results) return;
    setLoadingReport(true);

    try {
      const res = await fetch('/api/rag/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_info: patientInfo,
          multimodal_results: results,
          model_name: selectedModel
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to generate report');
      }

      const data = await res.json();
      setReportData(data);
    } catch (err) {
      alert(`RAG Report Error: ${err.message}`);
    } finally {
      setLoadingReport(false);
    }
  };

  // Interactive Clinical Chat
  const handleSendChat = async (caseContext, question) => {
    setChatLoading(true);
    try {
      const res = await fetch('/api/rag/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_context: caseContext,
          question: question,
          model_name: selectedModel
        })
      });

      if (!res.ok) throw new Error('Chat failed');
      const data = await res.json();
      return data.answer;
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation */}
      <Navbar
        systemStatus={systemStatus}
        onOpenAnalytics={() => setAnalyticsOpen(true)}
      />

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-5">
        {/* Workflow View Mode Switcher */}
        <div className="flex items-center justify-between border-b border-slate-900 pb-3">
          <div className="flex items-center space-x-2 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs font-semibold">
            <button
              onClick={() => setActiveWorkflowTab('all')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition ${
                activeWorkflowTab === 'all'
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <LayoutDashboard className="w-3.5 h-3.5" />
              <span>Full Clinical Cockpit</span>
            </button>

            <button
              onClick={() => setActiveWorkflowTab('imaging')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition ${
                activeWorkflowTab === 'imaging'
                  ? 'bg-cyan-600 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Imaging &amp; Calipers</span>
            </button>

            <button
              onClick={() => setActiveWorkflowTab('calculators')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition ${
                activeWorkflowTab === 'calculators'
                  ? 'bg-purple-600 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Calculator className="w-3.5 h-3.5" />
              <span>RTOG RPA Calculators</span>
            </button>

            <button
              onClick={() => setActiveWorkflowTab('rag')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition ${
                activeWorkflowTab === 'rag'
                  ? 'bg-amber-600 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>WHO CNS5 RAG Advisor</span>
            </button>
          </div>

          <div className="hidden sm:flex items-center space-x-2 text-xs text-slate-400 font-mono">
            <span>Model:</span>
            <span className="text-emerald-400 font-bold">95.75% Test Acc</span>
            <span className="text-slate-600">•</span>
            <span className="text-cyan-400 font-bold">100% Neoplasm Det</span>
          </div>
        </div>

        {/* Clinical Grid Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Patient Case & Biomarker Selector (5 Cols) */}
          <div className="lg:col-span-5 space-y-6">
            <PatientCaseSelector
              cases={cases}
              selectedCaseId={selectedCaseId}
              onSelectCase={handleSelectCase}
              patientInfo={patientInfo}
              onUpdatePatientInfo={setPatientInfo}
              onCustomImageUpload={handleCustomImageUpload}
              onRunAnalysis={handleRunAnalysis}
              loading={loadingAnalysis}
            />

            {/* In 'all' or 'calculators' tab, render Calculators in left or bottom column */}
            {(activeWorkflowTab === 'all' || activeWorkflowTab === 'calculators') && (
              <ClinicalCalculators
                patientInfo={patientInfo}
                results={results}
              />
            )}
          </div>

          {/* Right Column: Visualizer, Diagnostics & RAG Suite (7 Cols) */}
          <div className="lg:col-span-7 space-y-6">
            {/* Interactive MRI Visualizer (Shown in 'all' and 'imaging') */}
            {(activeWorkflowTab === 'all' || activeWorkflowTab === 'imaging') && (
              <MriViewer
                originalImage={originalImage}
                camOverlay={camOverlay}
                gnnOverlay={gnnOverlay}
                segOverlay={segOverlay}
                results={results}
              />
            )}

            {/* Diagnostic Classification & Multi-Task Results (Shown in 'all' and 'imaging') */}
            {(activeWorkflowTab === 'all' || activeWorkflowTab === 'imaging') && (
              <DiagnosticResults results={results} />
            )}

            {/* In 'calculators' isolated view, also show diagnostic results for context */}
            {activeWorkflowTab === 'calculators' && (
              <DiagnosticResults results={results} />
            )}

            {/* RAG Clinical Consultation & Evidence Drawer (Shown in 'all' and 'rag') */}
            {(activeWorkflowTab === 'all' || activeWorkflowTab === 'rag') && (
              <RagConsultation
                results={results}
                patientInfo={patientInfo}
                availableModels={systemStatus?.models || [selectedModel]}
                selectedModel={selectedModel}
                onSelectModel={setSelectedModel}
                onGenerateReport={handleGenerateReport}
                reportData={reportData}
                loadingReport={loadingReport}
                onSendChat={handleSendChat}
                chatLoading={chatLoading}
              />
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-500">
        NeuroMultimodal AI • Deep CNN (ResNet18) + Graph Attention Network (GATv2) + Multi-Head Cross-Attention + Clinical RAG ({selectedModel})
      </footer>

      {/* Analytics Modal */}
      <AnalyticsModal
        isOpen={analyticsOpen}
        onClose={() => setAnalyticsOpen(false)}
        metricsData={metricsData}
      />
    </div>
  );
}
