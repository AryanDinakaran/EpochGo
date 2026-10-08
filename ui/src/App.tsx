import React, { useState, useEffect } from 'react';
import {
  Sun,
  Moon,
  Plus,
  Play,
  RotateCw,
  Trash2,
  FileCode2,
  FileText,
  Cpu,
  Trophy,
  Box,
  Server,
  Download,
  UploadCloud,
  X,
  ArrowRight,
  Layers,
  Loader2,
  Check,
  AlertTriangle,
} from 'lucide-react';
import { marked } from 'marked';

interface LogItem {
  timestamp: string;
  step: string;
  message: string;
  phase?: string;
}

interface Project {
  id: string;
  name: string;
  status: string;
  target_column?: string;
  task_type?: string;
  rows?: number;
  num_columns?: number;
  best_model?: string;
  logs?: LogItem[];
  error?: string;
}

interface Feature {
  name: string;
  type: string;
  sample: any;
  is_numeric: boolean;
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [isDark, setIsDark] = useState(true);
  const [activeModel, setActiveModel] = useState('qwen2.5-coder:7b');
  const [liveLog, setLiveLog] = useState<string>('');

  // Modal states
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [previewFile, setPreviewFile] = useState<{ filename: string; content: string; isMd: boolean } | null>(null);

  // Form states
  const [newProjName, setNewProjName] = useState('');
  const [newTargetCol, setNewTargetCol] = useState('');
  const [newTaskType, setNewTaskType] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Testing Interface
  const [features, setFeatures] = useState<Feature[]>([]);
  const [formInputs, setFormInputs] = useState<Record<string, any>>({});
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isRerunning, setIsRerunning] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem('epochgo_theme') || localStorage.getItem('slash_ai_theme');
    const prefersDark = savedTheme === 'dark' || (!savedTheme && true);
    setIsDark(prefersDark);
    if (prefersDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }

    fetchHealth();
    loadProjects();
  }, []);

  const toggleTheme = () => {
    const nextTheme = !isDark;
    setIsDark(nextTheme);
    localStorage.setItem('epochgo_theme', nextTheme ? 'dark' : 'light');
    if (nextTheme) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  useEffect(() => {
    if (selectedProjectId) {
      loadProjectDetails(selectedProjectId);
    } else {
      setProject(null);
    }
  }, [selectedProjectId]);

  // WebSocket for live agent logs and status changes
  useEffect(() => {
    if (!selectedProjectId) return;

    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let ws: WebSocket | null = null;
    let pingInterval: any = null;

    try {
      ws = new WebSocket(`${proto}//${window.location.host}/ws/projects/${selectedProjectId}`);
      ws.onopen = () => {
        pingInterval = setInterval(() => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            try { ws.send('ping'); } catch (e) {}
          }
        }, 20000);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'log') {
            setLiveLog(msg.message);
          } else if (msg.type === 'status') {
            loadProjects();
            loadProjectDetails(selectedProjectId);
          }
        } catch (e) {
          console.error(e);
        }
      };
    } catch (e) {
      console.error(e);
    }

    return () => {
      if (pingInterval) clearInterval(pingInterval);
      if (ws) ws.close();
    };
  }, [selectedProjectId]);

  // Fallback polling while project is actively in-progress (1.5s responsive heartbeat)
  useEffect(() => {
    if (!selectedProjectId || !project) return;
    const inProgress = ['created', 'preprocessing', 'model_building', 'serving'].includes(project.status);
    if (!inProgress) return;

    const interval = setInterval(() => {
      loadProjectDetails(selectedProjectId);
      loadProjects();
    }, 1500);

    return () => clearInterval(interval);
  }, [selectedProjectId, project?.status]);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      if (data.active_model) setActiveModel(data.active_model);
    } catch (e) {
      console.error(e);
    }
  };

  const loadProjects = async () => {
    try {
      const res = await fetch('/api/projects');
      const data = await res.json();
      setProjects(data);
    } catch (e) {
      console.error(e);
    }
  };

  const loadProjectDetails = async (id: string) => {
    try {
      const res = await fetch(`/api/projects/${id}`);
      const data = await res.json();
      setProject(data);

      if (data.status === 'completed') {
        const schemaRes = await fetch(`/api/projects/${id}/schema`);
        if (schemaRes.ok) {
          const schema = await schemaRes.json();
          if (schema && schema.feature_schema) {
            setFeatures(schema.feature_schema);
            const initialForm: Record<string, any> = {};
            schema.feature_schema.forEach((f: Feature) => {
              initialForm[f.name] = f.sample;
            });
            setFormInputs(initialForm);
          }
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return alert('Please select a CSV file');

    setIsSubmitting(true);
    const formData = new FormData();
    formData.append('name', newProjName);
    formData.append('file', selectedFile);
    if (newTargetCol) formData.append('target_column', newTargetCol);
    if (newTaskType) formData.append('task_type', newTaskType);

    try {
      const res = await fetch('/api/projects', { method: 'POST', body: formData });
      const created = await res.json();
      setIsModalOpen(false);
      setNewProjName('');
      setSelectedFile(null);
      await loadProjects();
      setSelectedProjectId(created.id);
    } catch (err: any) {
      alert('Failed: ' + err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRerun = async () => {
    if (!selectedProjectId || isInProgress || isRerunning) return;
    setIsRerunning(true);
    try {
      const res = await fetch(`/api/projects/${selectedProjectId}/run`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || 'Could not re-run pipeline');
      }
      await loadProjectDetails(selectedProjectId);
      await loadProjects();
    } catch (e: any) {
      alert('Failed to rerun: ' + e.message);
    } finally {
      setIsRerunning(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedProjectId || !confirm('Delete this project and its artifacts?')) return;
    try {
      await fetch(`/api/projects/${selectedProjectId}`, { method: 'DELETE' });
      setSelectedProjectId(null);
      setProject(null);
      await loadProjects();
    } catch (e: any) {
      alert('Delete failed: ' + e.message);
    }
  };

  const handlePreview = async (filename: string) => {
    if (!selectedProjectId) return;
    try {
      const res = await fetch(`/api/projects/${selectedProjectId}/files/${filename}`);
      if (!res.ok) throw new Error('File not generated yet');
      const data = await res.json();
      setPreviewFile({
        filename,
        content: data.content,
        isMd: filename.endsWith('.md'),
      });
    } catch (e: any) {
      alert('File is being generated by the agent pipeline...');
    }
  };

  const executePrediction = async () => {
    if (!selectedProjectId) return;
    setIsPredicting(true);
    try {
      const payload: Record<string, any> = {};
      features.forEach(f => {
        payload[f.name] = f.is_numeric ? parseFloat(formInputs[f.name] || 0) : formInputs[f.name];
      });

      const res = await fetch(`/api/projects/${selectedProjectId}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [payload] }),
      });
      const data = await res.json();
      if (data.predictions && data.predictions[0]) {
        setPredictionResult(data.predictions[0]);
      }
    } catch (e: any) {
      alert('Prediction failed: ' + e.message);
    } finally {
      setIsPredicting(false);
    }
  };

  const renderCardBadge = (status: string) => {
    switch (status) {
      case 'preprocessing':
        return (
          <span className="text-[10px] px-2.5 py-0.5 rounded-full font-semibold bg-amber-500/10 text-amber-500 border border-amber-500/20 flex items-center space-x-1">
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
            <span>Preprocessing</span>
          </span>
        );
      case 'model_building':
        return (
          <span className="text-[10px] px-2.5 py-0.5 rounded-full font-semibold bg-indigo-500/10 text-indigo-500 border border-indigo-500/20 flex items-center space-x-1">
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
            <span>Evaluating</span>
          </span>
        );
      case 'serving':
        return (
          <span className="text-[10px] px-2.5 py-0.5 rounded-full font-semibold bg-purple-500/10 text-purple-500 border border-purple-500/20 flex items-center space-x-1">
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
            <span>Deploying</span>
          </span>
        );
      case 'completed':
        return (
          <span className="text-[10px] px-2.5 py-0.5 rounded-full font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
            Ready
          </span>
        );
      case 'failed':
        return (
          <span className="text-[10px] px-2.5 py-0.5 rounded-full font-semibold bg-rose-500/10 text-rose-500 border border-rose-500/20">
            Failed
          </span>
        );
      default:
        return (
          <span className="text-[10px] px-2.5 py-0.5 rounded-full font-semibold bg-neutral-200 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400">
            Queued
          </span>
        );
    }
  };

  const renderOverviewStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return (
          <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
            Ready
          </span>
        );
      case 'failed':
        return (
          <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-rose-500/10 text-rose-500 border border-rose-500/20">
            Failed
          </span>
        );
      case 'preprocessing':
        return (
          <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-amber-500/10 text-amber-500 border border-amber-500/20 flex items-center space-x-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span>PREPROCESSING</span>
          </span>
        );
      case 'model_building':
        return (
          <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-indigo-500/10 text-indigo-500 border border-indigo-500/20 flex items-center space-x-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span>EVALUATING</span>
          </span>
        );
      case 'serving':
        return (
          <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-purple-500/10 text-purple-500 border border-purple-500/20 flex items-center space-x-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span>DEPLOYING</span>
          </span>
        );
      default:
        return (
          <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-neutral-200 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400">
            QUEUED
          </span>
        );
    }
  };

  const isInProgress = project && ['created', 'preprocessing', 'model_building', 'serving'].includes(project.status);
  const isCompleted = project && project.status === 'completed';
  const isFailed = project && project.status === 'failed';

  return (
    <div className="flex flex-col h-screen bg-[#f5f5f7] dark:bg-black text-neutral-900 dark:text-neutral-100 font-sans transition-colors duration-200">
      {/* ================= STICKY TOP BAR ================= */}
      <header className="sticky top-0 z-40 border-b border-[#e5e5ea] dark:border-[#27272a] bg-white/70 dark:bg-black/70 backdrop-blur-xl px-6 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-2 cursor-pointer" onClick={() => setSelectedProjectId(null)}>
          <span className="text-xl font-bold tracking-tight text-neutral-950 dark:text-white">
            Epoch<span className="text-[#0071e3] font-extrabold">Go</span>
          </span>
          <span className="text-[11px] font-semibold tracking-wide text-neutral-500 dark:text-neutral-400 uppercase px-2.5 py-0.5 rounded-full bg-neutral-200/60 dark:bg-neutral-800/80">
            Epochlypse Research
          </span>
        </div>

        <div className="flex items-center space-x-3">
          <div className="hidden sm:flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-medium text-neutral-600 dark:text-neutral-300 bg-neutral-200/50 dark:bg-neutral-800/60 border border-neutral-300/40 dark:border-neutral-700/50">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="font-mono text-[11px]">{activeModel}</span>
          </div>

          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="w-9 h-9 rounded-full flex items-center justify-center bg-neutral-200/60 hover:bg-neutral-200 dark:bg-neutral-800/80 dark:hover:bg-neutral-700 text-neutral-700 dark:text-neutral-200 transition"
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </header>

      {/* ================= MAIN CONTAINER ================= */}
      <div className="flex flex-1 overflow-hidden">
        {/* ================= LEFT PANE: PROJECT CARDS ================= */}
        <aside className="w-80 md:w-96 border-r border-[#e5e5ea] dark:border-[#27272a] bg-neutral-100/40 dark:bg-neutral-950/40 flex flex-col shrink-0 overflow-hidden">
          <div className="px-5 py-4 border-b border-[#e5e5ea] dark:border-[#27272a] flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">Projects</span>
            <button onClick={loadProjects} className="text-neutral-400 hover:text-neutral-900 dark:hover:text-white transition p-1 rounded-lg">
              <RotateCw className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {!projects.length ? (
              <div className="text-center py-12 px-4 space-y-3">
                <p className="text-xs text-neutral-400">No projects yet.</p>
                <button onClick={() => setIsModalOpen(true)} className="text-xs text-[#0071e3] font-semibold hover:underline">
                  Create first project
                </button>
              </div>
            ) : (
              projects.map(p => {
                const isSelected = p.id === selectedProjectId;
                return (
                  <div
                    key={p.id}
                    onClick={() => setSelectedProjectId(p.id)}
                    className={`p-4 rounded-2xl cursor-pointer transition shadow-sm ${
                      isSelected
                        ? 'border-2 border-[#0071e3] bg-white dark:bg-[#161618] ring-2 ring-[#0071e3]/20'
                        : 'border border-[#e5e5ea] dark:border-[#27272a] bg-white dark:bg-[#161618] hover:border-neutral-400/50 dark:hover:border-neutral-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-bold text-neutral-950 dark:text-white truncate max-w-[170px]">{p.name}</span>
                      {renderCardBadge(p.status)}
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-neutral-500 dark:text-neutral-400">
                      <span>Target: <strong className="text-neutral-700 dark:text-neutral-200">{p.target_column || '-'}</strong></span>
                      <span>{p.rows || 0} rows</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        {/* ================= RIGHT PANE: ACTIVE PROJECT OR EMPTY ================= */}
        <main className="flex-1 flex flex-col overflow-y-auto bg-[#f5f5f7] dark:bg-black p-6 md:p-8">
          {!project ? (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center min-h-[500px]">
              <div className="w-20 h-20 rounded-3xl bg-white dark:bg-[#161618] border border-[#e5e5ea] dark:border-[#27272a] shadow-lg flex items-center justify-center text-[#0071e3] mb-5">
                <Layers className="w-9 h-9" />
              </div>
              <h2 className="text-2xl font-bold tracking-tight text-neutral-900 dark:text-white mb-2">Machine Learning, Simplified.</h2>
              <p className="text-neutral-500 dark:text-neutral-400 text-sm max-w-md mb-8 leading-relaxed">
                Select a project on the left, or create a new project to run your tabular CSV dataset through the 3 autonomous agents.
              </p>
              <button
                onClick={() => setIsModalOpen(true)}
                className="flex items-center space-x-2 bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-100 text-white dark:text-neutral-950 font-medium text-sm px-6 py-3 rounded-full shadow transition"
              >
                <Plus className="w-4 h-4" />
                <span>New Project</span>
              </button>
            </div>
          ) : (
            <div className="max-w-5xl mx-auto w-full space-y-8">
              {/* Top Overview Card */}
              <div className="bg-white dark:bg-[#161618] rounded-2xl p-6 shadow-sm border border-[#e5e5ea] dark:border-[#27272a] flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <div className="flex items-center space-x-3 mb-2">
                    <h1 className="text-2xl font-bold tracking-tight text-neutral-950 dark:text-white">{project.name}</h1>
                    {renderOverviewStatusBadge(project.status)}
                  </div>
                  <div className="flex flex-wrap items-center gap-3 text-xs text-neutral-500 dark:text-neutral-400">
                    <span>Target: <strong className="text-neutral-700 dark:text-neutral-300">{project.target_column || '-'}</strong></span>
                    <span>•</span>
                    <span>Task: <strong className="text-neutral-700 dark:text-neutral-300">{(project.task_type || '-').toUpperCase()}</strong></span>
                    <span>•</span>
                    <span>Dataset: <strong className="text-neutral-700 dark:text-neutral-300">{project.rows || '-'}</strong> rows, <strong className="text-neutral-700 dark:text-neutral-300">{project.num_columns || '-'}</strong> cols</span>
                  </div>
                </div>

                <div className="flex items-center space-x-2 shrink-0">
                  <button
                    onClick={handleRerun}
                    disabled={Boolean(isInProgress || isRerunning)}
                    className={`flex items-center space-x-1.5 text-xs font-medium px-3.5 py-2 rounded-xl transition ${
                      isInProgress || isRerunning
                        ? 'opacity-50 cursor-not-allowed pointer-events-none bg-neutral-100 dark:bg-neutral-800 text-neutral-400 dark:text-neutral-500'
                        : 'bg-neutral-100 hover:bg-neutral-200 dark:bg-neutral-800 dark:hover:bg-neutral-700 text-neutral-800 dark:text-neutral-200'
                    }`}
                  >
                    {isInProgress || isRerunning ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Agents Running...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-3.5 h-3.5 text-emerald-500" />
                        <span>Re-run Agents</span>
                      </>
                    )}
                  </button>
                  <button
                    onClick={() => setIsModalOpen(true)}
                    className="flex items-center space-x-1.5 text-xs font-medium px-3.5 py-2 rounded-xl bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-100 text-white dark:text-neutral-950 transition"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>New Project</span>
                  </button>
                  <button
                    onClick={handleDelete}
                    className="text-neutral-400 hover:text-rose-500 p-2 rounded-xl transition"
                    title="Delete Project"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* ================= IN-PROGRESS CENTRAL SCREEN ================= */}
              {isInProgress && (
                <div className="flex-1 flex flex-col items-center justify-center p-6 text-center min-h-[460px]">
                  <div className="bg-white dark:bg-[#161618] rounded-3xl p-8 shadow-sm border border-[#e5e5ea] dark:border-[#27272a] max-w-lg w-full flex flex-col items-center space-y-6">
                    {/* Animated Rotating Apple Loader */}
                    <div className="relative w-20 h-20 flex items-center justify-center">
                      <div className="absolute inset-0 rounded-full bg-[#0071e3]/10 dark:bg-[#0071e3]/20 animate-ping" />
                      <div className="w-16 h-16 rounded-full bg-white dark:bg-[#161618] border border-[#e5e5ea] dark:border-[#27272a] shadow-lg flex items-center justify-center text-[#0071e3] z-10">
                        <Loader2 className="w-8 h-8 animate-spin" />
                      </div>
                    </div>

                    {/* Phase Title & Description */}
                    <div className="space-y-1">
                      <h3 className="text-xl font-bold tracking-tight text-neutral-900 dark:text-white">
                        {project.status === 'preprocessing'
                          ? '1. Preprocessing Dataset'
                          : project.status === 'model_building'
                          ? '2. Evaluating Models & Tuning'
                          : project.status === 'serving'
                          ? '3. Synthesizing FastAPI Service'
                          : 'Synthesizing Machine Learning Pipeline'}
                      </h3>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400">
                        {project.status === 'preprocessing'
                          ? 'Agent 1 is profiling missing values, clipping outliers, and generating reproducible Scikit-Learn pipelines.'
                          : project.status === 'model_building'
                          ? 'Agent 2 is benchmarking 7 baseline algorithms (XGBoost, CatBoost, LightGBM) and tuning champion parameters.'
                          : project.status === 'serving'
                          ? 'Agent 3 is packaging the champion model into serve.py and spinning up the interactive prediction playground.'
                          : 'Our 3 autonomous agents are currently cleaning, benchmarking, and preparing your model.'}
                      </p>
                    </div>

                    {/* 3-Step Live Progress Checklist */}
                    <div className="w-full space-y-2.5 pt-2 text-left">
                      {/* Step 1 */}
                      <div className="p-3 rounded-2xl border border-[#e5e5ea] dark:border-[#27272a] bg-neutral-50/60 dark:bg-neutral-900/60 flex items-center space-x-3 transition">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 ${
                          ['model_building', 'serving', 'completed'].includes(project.status)
                            ? 'bg-emerald-500 text-white'
                            : 'bg-[#0071e3] text-white'
                        }`}>
                          {['model_building', 'serving', 'completed'].includes(project.status) ? (
                            <Check className="w-3.5 h-3.5" />
                          ) : (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          )}
                        </div>
                        <div className="truncate">
                          <div className="text-xs font-semibold text-neutral-900 dark:text-white">1. Data Hygiene & Preprocessing</div>
                          <div className="text-[11px] text-neutral-500 dark:text-neutral-400">Resolving missing values, IQR outlier treatment, scaling & encoding</div>
                        </div>
                      </div>

                      {/* Step 2 */}
                      <div className={`p-3 rounded-2xl border border-[#e5e5ea] dark:border-[#27272a] bg-neutral-50/60 dark:bg-neutral-900/60 flex items-center space-x-3 transition ${
                        !['model_building', 'serving', 'completed'].includes(project.status) ? 'opacity-50' : ''
                      }`}>
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 ${
                          ['serving', 'completed'].includes(project.status)
                            ? 'bg-emerald-500 text-white'
                            : project.status === 'model_building'
                            ? 'bg-[#0071e3] text-white'
                            : 'bg-neutral-200 dark:bg-neutral-800 text-neutral-500'
                        }`}>
                          {['serving', 'completed'].includes(project.status) ? (
                            <Check className="w-3.5 h-3.5" />
                          ) : project.status === 'model_building' ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            '2'
                          )}
                        </div>
                        <div className="truncate">
                          <div className="text-xs font-semibold text-neutral-900 dark:text-white">2. Model Benchmarks & Tuning</div>
                          <div className="text-[11px] text-neutral-500 dark:text-neutral-400">Evaluating 7 baseline algorithms & tuning champion parameters</div>
                        </div>
                      </div>

                      {/* Step 3 */}
                      <div className={`p-3 rounded-2xl border border-[#e5e5ea] dark:border-[#27272a] bg-neutral-50/60 dark:bg-neutral-900/60 flex items-center space-x-3 transition ${
                        !['serving', 'completed'].includes(project.status) ? 'opacity-50' : ''
                      }`}>
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 ${
                          project.status === 'completed'
                            ? 'bg-emerald-500 text-white'
                            : project.status === 'serving'
                            ? 'bg-[#0071e3] text-white'
                            : 'bg-neutral-200 dark:bg-neutral-800 text-neutral-500'
                        }`}>
                          {project.status === 'completed' ? (
                            <Check className="w-3.5 h-3.5" />
                          ) : project.status === 'serving' ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            '3'
                          )}
                        </div>
                        <div className="truncate">
                          <div className="text-xs font-semibold text-neutral-900 dark:text-white">3. API Deployment & Serving</div>
                          <div className="text-[11px] text-neutral-500 dark:text-neutral-400">Synthesizing standalone serve.py & interactive testing playground</div>
                        </div>
                      </div>
                    </div>

                    {/* Live Agent Thought Ticker */}
                    <div className="w-full p-3 bg-neutral-100/80 dark:bg-neutral-900/80 rounded-xl border border-[#e5e5ea] dark:border-[#27272a] flex items-center space-x-2 text-left">
                      <span className="w-2 h-2 rounded-full bg-[#0071e3] animate-pulse shrink-0"></span>
                      <span className="text-[11px] text-neutral-600 dark:text-neutral-300 font-mono truncate">
                        {liveLog || (project.logs && project.logs.length > 0 ? project.logs[project.logs.length - 1].message : 'Initializing autonomous agent crew...')}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* ================= FAILED ERROR SCREEN ================= */}
              {isFailed && (
                <div className="flex-1 flex flex-col items-center justify-center p-6 text-center min-h-[400px]">
                  <div className="bg-white dark:bg-[#161618] rounded-3xl p-8 shadow-sm border border-rose-500/20 max-w-md w-full flex flex-col items-center space-y-4">
                    <div className="w-14 h-14 rounded-full bg-rose-500/10 text-rose-500 flex items-center justify-center mb-1">
                      <AlertTriangle className="w-7 h-7" />
                    </div>
                    <h3 className="text-lg font-bold text-neutral-900 dark:text-white">Pipeline Execution Halted</h3>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400 leading-relaxed">
                      {project.error || 'An error occurred while running the agent crew.'}
                    </p>
                    <button
                      onClick={handleRerun}
                      className="mt-2 text-xs font-semibold px-5 py-2.5 rounded-full bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-100 text-white dark:text-neutral-950 transition"
                    >
                      Retry Pipeline
                    </button>
                  </div>
                </div>
              )}

              {/* ================= READY VIEW: 3 BANNERS (ONLY SHOWN WHEN COMPLETED) ================= */}
              {isCompleted && (
                <div className="space-y-8">
                  {/* ================= BANNER 1: PROCESSING ================= */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between px-1">
                      <h2 className="text-lg font-bold tracking-tight text-neutral-950 dark:text-white">Processing</h2>
                      <span className="text-xs text-neutral-400 font-medium">Agent 1: Data Preprocessor</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                      <div className="bg-white dark:bg-[#161618] rounded-2xl p-4 shadow-sm border border-[#e5e5ea] dark:border-[#27272a] flex items-center justify-between">
                        <div className="flex items-center space-x-3 truncate mr-2">
                          <div className="w-9 h-9 rounded-xl bg-blue-500/10 text-[#0071e3] flex items-center justify-center shrink-0">
                            <FileCode2 className="w-4 h-4" />
                          </div>
                          <div className="truncate">
                            <div className="text-sm font-semibold truncate">preprocessing.py</div>
                            <div className="text-[11px] text-neutral-400">Scikit-Learn Pipeline Script</div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-1">
                          <button onClick={() => handlePreview('preprocessing.py')} className="text-xs px-2.5 py-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800">
                            Preview
                          </button>
                          <a href={`/api/projects/${project.id}/download/preprocessing.py`} className="p-1.5 text-neutral-400 hover:text-[#0071e3]">
                            <Download className="w-4 h-4" />
                          </a>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-[#161618] rounded-2xl p-4 shadow-sm border border-[#e5e5ea] dark:border-[#27272a] flex items-center justify-between">
                        <div className="flex items-center space-x-3 truncate mr-2">
                          <div className="w-9 h-9 rounded-xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center shrink-0">
                            <FileText className="w-4 h-4" />
                          </div>
                          <div className="truncate">
                            <div className="text-sm font-semibold truncate">preprocessing.md</div>
                            <div className="text-[11px] text-neutral-400">Data Profiling & Hygiene Report</div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-1">
                          <button onClick={() => handlePreview('preprocessing.md')} className="text-xs px-2.5 py-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800">
                            Preview
                          </button>
                          <a href={`/api/projects/${project.id}/download/preprocessing.md`} className="p-1.5 text-neutral-400 hover:text-emerald-500">
                            <Download className="w-4 h-4" />
                          </a>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* ================= BANNER 2: MODEL BUILDING ================= */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between px-1">
                      <h2 className="text-lg font-bold tracking-tight text-neutral-950 dark:text-white">Model Building</h2>
                      <div className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-indigo-500/10 text-indigo-500 border border-indigo-500/20">
                        {project.best_model ? `Champion: ${project.best_model}` : '7 Benchmark Models'}
                      </div>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
                      <div className="bg-white dark:bg-[#161618] rounded-2xl p-4 shadow-sm border border-[#e5e5ea] dark:border-[#27272a] flex items-center justify-between">
                        <div className="flex items-center space-x-3 truncate mr-2">
                          <div className="w-9 h-9 rounded-xl bg-violet-500/10 text-violet-500 flex items-center justify-center shrink-0">
                            <Cpu className="w-4 h-4" />
                          </div>
                          <div className="truncate">
                            <div className="text-sm font-semibold truncate">model_building.py</div>
                            <div className="text-[11px] text-neutral-400">Training Script</div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-1">
                          <button onClick={() => handlePreview('model_building.py')} className="text-xs px-2.5 py-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800">
                            Preview
                          </button>
                          <a href={`/api/projects/${project.id}/download/model_building.py`} className="p-1.5 text-neutral-400 hover:text-violet-500">
                            <Download className="w-4 h-4" />
                          </a>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-[#161618] rounded-2xl p-4 shadow-sm border border-[#e5e5ea] dark:border-[#27272a] flex items-center justify-between">
                        <div className="flex items-center space-x-3 truncate mr-2">
                          <div className="w-9 h-9 rounded-xl bg-amber-500/10 text-amber-500 flex items-center justify-center shrink-0">
                            <Trophy className="w-4 h-4" />
                          </div>
                          <div className="truncate">
                            <div className="text-sm font-semibold truncate">model_building.md</div>
                            <div className="text-[11px] text-neutral-400">Benchmark Leaderboard & Stats</div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-1">
                          <button onClick={() => handlePreview('model_building.md')} className="text-xs px-2.5 py-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800">
                            Preview
                          </button>
                          <a href={`/api/projects/${project.id}/download/model_building.md`} className="p-1.5 text-neutral-400 hover:text-amber-500">
                            <Download className="w-4 h-4" />
                          </a>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-[#161618] rounded-2xl p-4 shadow-sm border border-[#e5e5ea] dark:border-[#27272a] flex items-center justify-between">
                        <div className="flex items-center space-x-3 truncate mr-2">
                          <div className="w-9 h-9 rounded-xl bg-indigo-500/10 text-indigo-500 flex items-center justify-center shrink-0">
                            <Box className="w-4 h-4" />
                          </div>
                          <div className="truncate">
                            <div className="text-sm font-semibold truncate">model.joblib</div>
                            <div className="text-[11px] text-neutral-400">Trained Champion Pipeline</div>
                          </div>
                        </div>
                        <a href={`/api/projects/${project.id}/download/model.joblib`} className="text-xs font-medium bg-neutral-100 hover:bg-neutral-200 dark:bg-neutral-800 dark:hover:bg-neutral-700 px-3 py-1.5 rounded-lg flex items-center space-x-1">
                          <Download className="w-3.5 h-3.5" />
                          <span>Get</span>
                        </a>
                      </div>
                    </div>
                  </div>

                  {/* ================= BANNER 3: SERVER ================= */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between px-1">
                      <h2 className="text-lg font-bold tracking-tight text-neutral-950 dark:text-white">Server</h2>
                      <span className="text-xs text-neutral-400 font-medium">FastAPI Endpoint & Playground</span>
                    </div>

                    <div className="bg-white dark:bg-[#161618] rounded-2xl p-4 shadow-sm border border-[#e5e5ea] dark:border-[#27272a] flex items-center justify-between">
                      <div className="flex items-center space-x-3 truncate mr-2">
                        <div className="w-9 h-9 rounded-xl bg-purple-500/10 text-purple-500 flex items-center justify-center shrink-0">
                          <Server className="w-4 h-4" />
                        </div>
                        <div className="truncate">
                          <div className="text-sm font-semibold truncate">serve.py</div>
                          <div className="text-[11px] text-neutral-400">Production Microservice Script</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <button onClick={() => handlePreview('serve.py')} className="text-xs px-2.5 py-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800">
                          Preview
                        </button>
                        <a href={`/api/projects/${project.id}/download/serve.py`} className="p-1.5 text-neutral-400 hover:text-purple-500">
                          <Download className="w-4 h-4" />
                        </a>
                      </div>
                    </div>

                    {/* Interactive Testing API */}
                    <div className="bg-white dark:bg-[#161618] rounded-2xl p-6 shadow-sm border border-[#e5e5ea] dark:border-[#27272a] space-y-6">
                      <div className="flex items-center justify-between border-b border-[#e5e5ea] dark:border-[#27272a] pb-4">
                        <div>
                          <h3 className="text-sm font-bold text-neutral-900 dark:text-white">Interactive Testing Interface</h3>
                          <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">Test live inference against the trained champion model.</p>
                        </div>
                        <button
                          onClick={executePrediction}
                          disabled={isPredicting}
                          className="bg-[#0071e3] hover:bg-[#0077ed] text-white text-xs font-semibold px-4 py-2 rounded-xl shadow transition flex items-center space-x-1.5"
                        >
                          <Play className="w-3.5 h-3.5" />
                          <span>{isPredicting ? 'Inferring...' : 'Run Prediction'}</span>
                        </button>
                      </div>

                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                          <label className="block text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-3">Input Features</label>
                          <div className="space-y-2.5 max-h-72 overflow-y-auto pr-2">
                            {features.map(f => (
                              <div key={f.name} className="flex items-center justify-between space-x-2">
                                <label className="text-xs font-medium text-neutral-600 dark:text-neutral-400 truncate w-36">{f.name}</label>
                                <input
                                  type={f.is_numeric ? 'number' : 'text'}
                                  step="any"
                                  value={formInputs[f.name] ?? ''}
                                  onChange={e => setFormInputs({ ...formInputs, [f.name]: e.target.value })}
                                  className="w-full max-w-[200px] bg-neutral-100 dark:bg-neutral-900 border border-[#e5e5ea] dark:border-[#27272a] rounded-lg px-2.5 py-1 text-xs text-neutral-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-[#0071e3]"
                                />
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="flex flex-col justify-between space-y-4">
                          <div>
                            <label className="block text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-3">Model Output</label>
                            <div className="bg-neutral-100/70 dark:bg-neutral-900/70 rounded-xl p-5 text-center flex flex-col items-center justify-center min-h-[120px] border border-[#e5e5ea] dark:border-[#27272a]">
                              {predictionResult ? (
                                <div>
                                  <div className="text-[10px] uppercase font-semibold text-neutral-400 tracking-wider mb-0.5">Predicted Value</div>
                                  <div className="text-3xl font-extrabold text-emerald-500">{predictionResult.prediction}</div>
                                  {predictionResult.probabilities && (
                                    <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                                      Confidence: [{predictionResult.probabilities.map((pr: number) => `${(pr * 100).toFixed(1)}%`).join(', ')}]
                                    </div>
                                  )}
                                </div>
                              ) : (
                                <span className="text-xs text-neutral-400">Click "Run Prediction" to test values</span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      {/* ================= "+ NEW PROJECT" FROSTED BLUR MODAL (CLICK OUTSIDE DISMISSES) ================= */}
      {isModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/30 dark:bg-black/60 backdrop-blur-md"
          onClick={e => {
            if (e.target === e.currentTarget) setIsModalOpen(false);
          }}
        >
          <div className="bg-white dark:bg-[#161618] border border-[#e5e5ea] dark:border-[#27272a] rounded-3xl w-full max-w-lg p-7 shadow-2xl space-y-5">
            <div className="flex items-center justify-between border-b border-[#e5e5ea] dark:border-[#27272a] pb-4">
              <div>
                <h3 className="text-lg font-bold tracking-tight text-neutral-900 dark:text-white">New Project</h3>
                <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">Upload a CSV dataset to initiate the 3 agents.</p>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="w-8 h-8 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-500 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white flex items-center justify-center transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreateProject} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-neutral-700 dark:text-neutral-300 mb-1.5">Project Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Customer Churn Predictor"
                  value={newProjName}
                  onChange={e => setNewProjName(e.target.value)}
                  className="w-full bg-neutral-100/80 dark:bg-neutral-900 border border-[#e5e5ea] dark:border-[#27272a] rounded-xl px-4 py-2.5 text-sm text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-[#0071e3] transition"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-neutral-700 dark:text-neutral-300 mb-1.5">CSV Dataset File</label>
                <div
                  className="border-2 border-dashed border-neutral-300 dark:border-neutral-700 hover:border-[#0071e3] bg-neutral-50 dark:bg-neutral-900/40 rounded-2xl p-6 text-center cursor-pointer transition"
                  onClick={() => document.getElementById('react-input-file')?.click()}
                >
                  <input
                    type="file"
                    id="react-input-file"
                    accept=".csv"
                    required
                    className="hidden"
                    onChange={e => setSelectedFile(e.target.files?.[0] || null)}
                  />
                  <UploadCloud className="w-8 h-8 text-[#0071e3] mx-auto mb-2" />
                  <p className="text-xs text-neutral-700 dark:text-neutral-300 font-medium">
                    {selectedFile ? `${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)` : 'Click to upload or drag & drop CSV file'}
                  </p>
                  <p className="text-[11px] text-neutral-400 mt-1">Raw tabular dataset in CSV format</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-neutral-700 dark:text-neutral-300 mb-1.5">
                    Target Column <span className="text-neutral-400 font-normal">(Optional)</span>
                  </label>
                  <input
                    type="text"
                    placeholder="Auto-inferred if empty"
                    value={newTargetCol}
                    onChange={e => setNewTargetCol(e.target.value)}
                    className="w-full bg-neutral-100/80 dark:bg-neutral-900 border border-[#e5e5ea] dark:border-[#27272a] rounded-xl px-3.5 py-2 text-xs text-neutral-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-[#0071e3]"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-neutral-700 dark:text-neutral-300 mb-1.5">Task Type</label>
                  <select
                    value={newTaskType}
                    onChange={e => setNewTaskType(e.target.value)}
                    className="w-full bg-neutral-100/80 dark:bg-neutral-900 border border-[#e5e5ea] dark:border-[#27272a] rounded-xl px-3.5 py-2 text-xs text-neutral-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-[#0071e3]"
                  >
                    <option value="">Auto-Detect</option>
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>
              </div>

              <div className="pt-3 flex items-center justify-between border-t border-[#e5e5ea] dark:border-[#27272a]">
                <div className="text-[10px] text-neutral-400 font-mono tracking-wider uppercase">Epochlypse Research &bull; Aryan Dinakaran</div>
                <div className="flex items-center space-x-3">
                  <button
                    type="button"
                    onClick={() => setIsModalOpen(false)}
                    className="px-4 py-2 text-xs font-medium text-neutral-500 hover:text-neutral-900 dark:hover:text-white transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="bg-[#0071e3] hover:bg-[#0077ed] text-white text-xs font-semibold px-5 py-2.5 rounded-full shadow transition flex items-center space-x-1.5"
                  >
                    <span>{isSubmitting ? 'Creating...' : 'Launch CrewAI'}</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ================= FILE PREVIEW MODAL ================= */}
      {previewFile && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 dark:bg-black/70 backdrop-blur-md"
          onClick={e => {
            if (e.target === e.currentTarget) setPreviewFile(null);
          }}
        >
          <div className="bg-white dark:bg-[#161618] border border-[#e5e5ea] dark:border-[#27272a] rounded-3xl w-full max-w-3xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-[#e5e5ea] dark:border-[#27272a] flex items-center justify-between shrink-0">
              <span className="text-sm font-bold text-neutral-900 dark:text-white font-mono">{previewFile.filename}</span>
              <button
                onClick={() => setPreviewFile(null)}
                className="w-8 h-8 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-500 hover:text-neutral-900 dark:hover:text-white flex items-center justify-center"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-6 overflow-y-auto flex-1 text-xs">
              {previewFile.isMd ? (
                <div
                  className="prose dark:prose-invert max-w-none"
                  dangerouslySetInnerHTML={{ __html: marked.parse(previewFile.content) }}
                />
              ) : (
                <pre className="bg-neutral-100 dark:bg-neutral-900 p-4 rounded-xl font-mono text-xs overflow-x-auto text-neutral-800 dark:text-neutral-200">
                  {previewFile.content}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
