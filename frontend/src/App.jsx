import React, { useState } from 'react';
import { Upload, CheckCircle, XCircle, FileText, Award, GraduationCap, Code, AlertCircle, Loader2, ExternalLink, ChevronRight, Trophy } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

export default function ResumeVerifier() {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [role, setRole] = useState('Python Developer');
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const API_URL = 'http://localhost:8000';

  const handleFileUpload = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError(null);
    } else {
      setError('Please upload a PDF file');
      setFile(null);
      setFileName('');
    }
  };

  const handleProcess = async () => {
    if (!file) {
      setError('Please upload a resume first');
      return;
    }

    setProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('role', role);

      const response = await fetch(`${API_URL}/api/verify-resume`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process resume');
      }

      const data = await response.json();

      if (data.success) {
        setResults(data);
        setActiveTab('achievements');
      } else {
        setError('Failed to process resume. Please try again.');
      }
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'Failed to connect to server. Make sure the backend is running on port 8000.');
    } finally {
      setProcessing(false);
    }
  };

  const getCategoryIcon = (category) => {
    switch (category) {
      case 'role_based': return <Code className="w-4 h-4" />;
      case 'project_specific': return <FileText className="w-4 h-4" />;
      case 'system_design': return <Trophy className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case 'role_based': return 'bg-blue-500';
      case 'project_specific': return 'bg-purple-500';
      case 'system_design': return 'bg-orange-500';
      default: return 'bg-gray-500';
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'publication': return <FileText className="w-5 h-5" />;
      case 'hackathon': return <Trophy className="w-5 h-5" />;
      case 'open_source': return <Code className="w-5 h-5" />;
      default: return <Award className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-3 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
            Resume Verification System
          </h1>
          <p className="text-gray-300 text-lg">Verify achievements, certifications, and generate project-based MCQs</p>
        </div>

        {/* Main Card */}
        <div className="bg-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-slate-700">
          {/* Tabs */}
          <div className="flex border-b border-slate-700 bg-slate-900/50">
            {['upload', 'achievements', 'certifications', 'mcqs'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                disabled={tab !== 'upload' && !results}
                className={`flex-1 px-6 py-4 text-sm font-medium transition-all ${activeTab === tab
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white border-b-2 border-white'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-slate-800'
                  } ${tab !== 'upload' && !results ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {tab === 'upload' && <Upload className="w-4 h-4 inline mr-2" />}
                {tab === 'achievements' && <Award className="w-4 h-4 inline mr-2" />}
                {tab === 'certifications' && <GraduationCap className="w-4 h-4 inline mr-2" />}
                {tab === 'mcqs' && <FileText className="w-4 h-4 inline mr-2" />}
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="p-8">
            {error && (
              <Alert className="mb-6 bg-red-900/50 border-red-600 text-red-200">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Upload Tab */}
            {activeTab === 'upload' && (
              <div className="space-y-6">
                <div className="border-2 border-dashed border-slate-600 rounded-xl p-12 text-center hover:border-purple-500 transition-colors bg-slate-900/30">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <Upload className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                    <p className="text-xl font-semibold text-white mb-2">Upload Resume (PDF)</p>
                    <p className="text-gray-400">Click to browse or drag and drop</p>
                  </label>
                  {fileName && (
                    <div className="mt-4 inline-flex items-center gap-2 bg-green-900/30 text-green-300 px-4 py-2 rounded-lg">
                      <CheckCircle className="w-5 h-5" />
                      {fileName}
                    </div>
                  )}
                </div>

                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
                  <label className="block text-sm font-medium text-gray-300 mb-3">
                    Target Role
                  </label>
                  <select
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option>Python Developer</option>
                    <option>Full Stack Developer</option>
                    <option>ML Engineer</option>
                    <option>Data Scientist</option>
                    <option>DevOps Engineer</option>
                  </select>
                </div>

                <button
                  onClick={handleProcess}
                  disabled={!file || processing}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 rounded-xl font-semibold text-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] flex items-center justify-center gap-2 shadow-lg"
                >
                  {processing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Processing Resume...
                    </>
                  ) : (
                    <>
                      Process Resume
                      <ChevronRight className="w-5 h-5" />
                    </>
                  )}
                </button>

                <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4 mt-6">
                  <p className="text-blue-300 text-sm">
                    <strong>Note:</strong> Make sure the FastAPI backend is running on port 8000. Run: <code className="bg-slate-800 px-2 py-1 rounded">python backend.py</code>
                  </p>
                </div>
              </div>
            )}

            {/* Achievements Tab */}
            {activeTab === 'achievements' && results && (
              <div className="space-y-4">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-white">Achievement Verification</h2>
                  <div className="text-sm text-gray-400">
                    {results.achievements.filter(a => a.verification?.verified).length} / {results.achievements.length} Verified
                  </div>
                </div>

                {results.achievements.length === 0 ? (
                  <div className="text-center py-12 text-gray-400">
                    <Award className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>No achievements found in resume</p>
                  </div>
                ) : (
                  results.achievements.map((achievement, idx) => (
                    <div key={idx} className="bg-slate-900/50 rounded-xl p-6 border border-slate-700 hover:border-slate-600 transition-colors">
                      <div className="flex items-start gap-4">
                        <div className={`p-3 rounded-lg ${achievement.verification?.verified ? 'bg-green-900/30' : 'bg-red-900/30'}`}>
                          {getTypeIcon(achievement.type)}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-start justify-between mb-2">
                            <h3 className="text-lg font-semibold text-white">{achievement.title}</h3>
                            {achievement.verification?.verified ? (
                              <CheckCircle className="w-5 h-5 text-green-400" />
                            ) : (
                              <XCircle className="w-5 h-5 text-red-400" />
                            )}
                          </div>
                          <div className="space-y-1 text-sm text-gray-400 mb-3">
                            <p><strong className="text-gray-300">Type:</strong> {achievement.type}</p>
                            <p><strong className="text-gray-300">Event:</strong> {achievement.event_name}</p>
                            <p><strong className="text-gray-300">Level:</strong> {achievement.level}</p>
                            <p><strong className="text-gray-300">Year:</strong> {achievement.year}</p>
                          </div>
                          {achievement.verification?.verified ? (
                            <div className="bg-green-900/20 border border-green-800 rounded-lg p-3">
                              <p className="text-green-300 text-sm font-medium mb-2">
                                ✓ Verified - Found {achievement.verification.results_found} results
                              </p>
                              {achievement.verification.top_results?.[0] && (
                                <a
                                  href={achievement.verification.top_results[0].url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1"
                                >
                                  {achievement.verification.top_results[0].title}
                                  <ExternalLink className="w-3 h-3" />
                                </a>
                              )}
                            </div>
                          ) : (
                            <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
                              <p className="text-red-300 text-sm">
                                ✗ {achievement.verification?.error || 'Not verified'}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Certifications Tab */}
            {activeTab === 'certifications' && results && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-white mb-6">Certifications</h2>

                {results.certifications.with_links.length === 0 && results.certifications.without_links.length === 0 ? (
                  <div className="text-center py-12 text-gray-400">
                    <GraduationCap className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>No certifications found in resume</p>
                  </div>
                ) : (
                  <>
                    {results.certifications.with_links.length > 0 && (
                      <div>
                        <div className="flex items-center gap-2 mb-4">
                          <CheckCircle className="w-5 h-5 text-green-400" />
                          <h3 className="text-lg font-semibold text-white">With Verification Links ({results.certifications.with_links.length})</h3>
                        </div>
                        <div className="space-y-3">
                          {results.certifications.with_links.map((cert, idx) => (
                            <div key={idx} className="bg-slate-900/50 rounded-lg p-4 border border-slate-700 hover:border-green-600 transition-colors">
                              <div className="flex items-start justify-between">
                                <div>
                                  <h4 className="font-semibold text-white mb-1">{cert.name}</h4>
                                  <p className="text-sm text-gray-400">{cert.issuer} {cert.year && `• ${cert.year}`}</p>
                                </div>
                                <a
                                  href={cert.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="bg-green-900/30 text-green-300 px-3 py-1 rounded-md text-sm hover:bg-green-900/50 transition-colors flex items-center gap-1 whitespace-nowrap"
                                >
                                  Verify
                                  <ExternalLink className="w-3 h-3" />
                                </a>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {results.certifications.without_links.length > 0 && (
                      <div>
                        <div className="flex items-center gap-2 mb-4">
                          <AlertCircle className="w-5 h-5 text-yellow-400" />
                          <h3 className="text-lg font-semibold text-white">Without Verification Links ({results.certifications.without_links.length})</h3>
                        </div>
                        <div className="space-y-3">
                          {results.certifications.without_links.map((cert, idx) => (
                            <div key={idx} className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                              <h4 className="font-semibold text-white mb-1">{cert.name}</h4>
                              <p className="text-sm text-gray-400">{cert.issuer} {cert.year && `• ${cert.year}`}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {/* MCQs Tab */}
            {activeTab === 'mcqs' && results && results.mcqs && (
              <div className="space-y-6">
                {!results.mcqs.success ? (
                  <div className="text-center py-12 text-gray-400">
                    <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>No projects found in resume to generate MCQs</p>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center justify-between mb-6">
                      <h2 className="text-2xl font-bold text-white">Generated MCQs</h2>
                      <div className="text-sm text-gray-400">
                        Role: <span className="text-purple-400 font-semibold">{results.mcqs.role}</span>
                      </div>
                    </div>

                    {/* Projects Summary */}
                    <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700 mb-6">
                      <h3 className="text-lg font-semibold text-white mb-4">Based on Projects:</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {results.mcqs.projects.map((project, idx) => (
                          <div key={idx} className="bg-slate-800 rounded-lg p-4 border border-slate-600">
                            <h4 className="font-semibold text-white mb-2">{project.title}</h4>
                            <p className="text-sm text-gray-400 mb-2">{project.description}</p>
                            <div className="flex flex-wrap gap-2">
                              {project.technologies.map((tech, i) => (
                                <span key={i} className="bg-purple-900/30 text-purple-300 text-xs px-2 py-1 rounded">
                                  {tech}
                                </span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Questions */}
                    <div className="space-y-4">
                      {results.mcqs.questions.map((q, idx) => (
                        <div key={idx} className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
                          <div className="flex items-start gap-3 mb-4">
                            <div className={`${getCategoryColor(q.category)} p-2 rounded-lg`}>
                              {getCategoryIcon(q.category)}
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-2">
                                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
                                  {q.category.replace('_', ' ')}
                                </span>
                                {q.related_project && (
                                  <span className="text-xs text-purple-400">• {q.related_project}</span>
                                )}
                              </div>
                              <h3 className="text-lg font-semibold text-white mb-4">
                                Q{idx + 1}. {q.question}
                              </h3>
                            </div>
                          </div>

                          <div className="space-y-2 mb-4">
                            {q.options.map((option, optIdx) => (
                              <div
                                key={optIdx}
                                className={`p-3 rounded-lg border transition-colors ${option === q.correct_answer
                                    ? 'bg-green-900/20 border-green-600 text-green-300'
                                    : 'bg-slate-800 border-slate-700 text-gray-300'
                                  }`}
                              >
                                {option === q.correct_answer && <span className="mr-2">✓</span>}
                                {option}
                              </div>
                            ))}
                          </div>

                          <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-3">
                            <p className="text-sm text-blue-300">
                              <strong>Explanation:</strong> {q.explanation}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-400 text-sm">
          <p>Resume Verification System • Powered by AI</p>
        </div>
      </div>
    </div>
  );
}