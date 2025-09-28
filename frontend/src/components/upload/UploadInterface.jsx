// Enhanced Upload Interface Component
// frontend/src/components/upload/UploadInterface.jsx
import React, { useState, useCallback } from 'react';
import FileDropzone from './FileDropzone';
import UploadProgress from './UploadProgress';
import InvoiceResponse from './InvoiceResponse';
import { Upload, FileText, TrendingUp } from 'lucide-react';

const UploadInterface = ({ onFileUpload, processing }) => {
  const [uploadState, setUploadState] = useState('idle'); // 'idle', 'uploading', 'processing', 'completed'
  const [uploads, setUploads] = useState([]);
  const [analysisResults, setAnalysisResults] = useState([]);

  const handleFileUpload = useCallback(async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    setUploadState('uploading');
    setUploads([]);
    setAnalysisResults([]);

    try {
      // Create initial upload tracking
      const initialUploads = files.map((file, index) => ({
        id: `upload_${Date.now()}_${index}`,
        fileName: file.name,
        status: 'uploading',
        progress: 0,
        currentStep: 'Uploading file...'
      }));
      setUploads(initialUploads);

      // Simulate realistic processing with 60-second timeline
      const startTime = Date.now();
      
      // Phase 1: Upload (0-10% in first 5 seconds)
      const uploadInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const uploadProgress = Math.min((elapsed / 5000) * 10, 10);
        
        setUploads(prev => prev.map(upload => ({
          ...upload,
          progress: uploadProgress,
          currentStep: uploadProgress < 10 ? 'Uploading file...' : 'Upload complete'
        })));

        if (uploadProgress >= 10) {
          clearInterval(uploadInterval);
          startProcessingPhase();
        }
      }, 200);

      const startProcessingPhase = () => {
        setUploadState('processing');
        
        // Phase 2: Processing (10-90% over 30 seconds, then 90-100% over remaining 25 seconds)
        const processingSteps = [
          { step: 'Extracting invoice data...', progress: 20, duration: 5000 },
          { step: 'Running fraud detection agents...', progress: 40, duration: 8000 },
          { step: 'Analyzing vendor information...', progress: 60, duration: 7000 },
          { step: 'Validating amounts and duplicates...', progress: 80, duration: 6000 },
          { step: 'Cross-referencing databases...', progress: 90, duration: 4000 }, // Reaches 90% at 30 seconds
          { step: 'Finalizing risk assessment...', progress: 95, duration: 15000 },
          { step: 'Generating report...', progress: 98, duration: 8000 },
          { step: 'Analysis complete!', progress: 100, duration: 2000 }
        ];

        let stepIndex = 0;
        let stepStartTime = Date.now();

        const processStep = () => {
          if (stepIndex >= processingSteps.length) {
            completeAnalysis();
            return;
          }

          const currentStep = processingSteps[stepIndex];
          const stepElapsed = Date.now() - stepStartTime;
          const stepProgress = Math.min((stepElapsed / currentStep.duration), 1);
          
          // Interpolate progress within the step
          const prevProgress = stepIndex > 0 ? processingSteps[stepIndex - 1].progress : 10;
          const currentProgress = prevProgress + (currentStep.progress - prevProgress) * stepProgress;

          setUploads(prev => prev.map(upload => ({
            ...upload,
            status: 'processing',
            progress: Math.round(currentProgress),
            currentStep: currentStep.step
          })));

          if (stepProgress >= 1) {
            stepIndex++;
            stepStartTime = Date.now();
          }

          if (stepIndex < processingSteps.length || stepProgress < 1) {
            setTimeout(processStep, 200);
          }
        };

        processStep();
      };

      const completeAnalysis = async () => {
        // Simulate API call for final results
        const mockAnalysisResult = {
          invoice_id: `inv_${Date.now()}`,
          overall_risk_score: Math.round(Math.random() * 40 + 10), // 10-50% risk
          confidence: 0.92 + Math.random() * 0.07, // 92-99% confidence
          status: Math.random() > 0.8 ? 'review_required' : 'approved',
          recommendation: Math.random() > 0.8 
            ? 'Manual review recommended due to detected anomalies' 
            : 'Invoice approved for processing',
          red_flags: Math.random() > 0.7 ? [
            'Vendor address mismatch detected',
            'Amount exceeds typical range for this vendor'
          ] : [],
          agent_results: [
            {
              agent_id: 'fraud_detector',
              status: 'completed',
              risk_score: Math.round(Math.random() * 30 + 5),
              analysis: 'No significant fraud indicators detected in invoice structure and content.',
              findings: ['Data structure verified', 'No anomalies detected'],
              execution_time: 3.2
            },
            {
              agent_id: 'vendor_validator',
              status: 'completed', 
              risk_score: Math.round(Math.random() * 25 + 10),
              analysis: 'Vendor verification completed successfully with minor discrepancies.',
              findings: ['Vendor registered', 'Address partially verified'],
              execution_time: 2.8
            },
            {
              agent_id: 'duplicate_scanner',
              status: 'completed',
              risk_score: Math.round(Math.random() * 15),
              analysis: 'No duplicate invoices found in system database.',
              findings: ['No duplicates found'],
              execution_time: 1.5
            },
            {
              agent_id: 'amount_validator',
              status: 'completed',
              risk_score: Math.round(Math.random() * 20 + 5),
              analysis: 'Amount validation completed within acceptable parameters.',
              findings: ['Amount within normal range'],
              execution_time: 2.1
            }
          ],
          analysis_summary: 'Invoice processing completed successfully. All fraud detection agents have analyzed the document and provided risk assessments.',
          processing_time: 58.4,
          metadata: {
            total_agents_executed: 4,
            successful_agents: 4,
            failed_agents: 0,
            unique_red_flags_count: Math.random() > 0.7 ? 2 : 0,
            analysis_timestamp: new Date().toISOString()
          }
        };

        // Update final status
        setUploads(prev => prev.map(upload => ({
          ...upload,
          status: 'completed',
          progress: 100,
          currentStep: 'Analysis complete!',
          processingTime: 58.4,
          result: {
            riskScore: mockAnalysisResult.overall_risk_score,
            confidence: Math.round(mockAnalysisResult.confidence * 100),
            status: mockAnalysisResult.status
          }
        })));

        setAnalysisResults([mockAnalysisResult]);
        setUploadState('completed');
      };

    } catch (error) {
      console.error('Upload failed:', error);
      setUploads(prev => prev.map(upload => ({
        ...upload,
        status: 'failed',
        error: error.message
      })));
      setUploadState('idle');
    }

    // Call parent handler if provided
    if (onFileUpload) {
      onFileUpload(event);
    }
  }, [onFileUpload]);

  const handleReset = () => {
    setUploadState('idle');
    setUploads([]);
    setAnalysisResults([]);
  };

  const handleViewResults = () => {
    setUploadState('completed');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Invoice Upload & Analysis</h2>
        <p className="text-gray-600">
          Upload your invoice files for automated fraud detection and risk analysis
        </p>
      </div>

      {/* Upload State Machine */}
      {uploadState === 'idle' && (
        <FileDropzone 
          onFileUpload={handleFileUpload} 
          processing={processing || uploadState !== 'idle'} 
        />
      )}

      {(uploadState === 'uploading' || uploadState === 'processing') && (
        <div className="space-y-4">
          <UploadProgress 
            uploads={uploads} 
            onComplete={handleViewResults}
          />
          
          {/* Processing Info */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <div>
                <h3 className="font-medium text-blue-900">
                  {uploadState === 'uploading' ? 'Uploading Files...' : 'Running Analysis...'}
                </h3>
                <p className="text-sm text-blue-700">
                  {uploadState === 'uploading' 
                    ? 'Transferring files to secure processing environment'
                    : 'Parallel fraud detection agents are analyzing your invoices'
                  }
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {uploadState === 'completed' && analysisResults.length > 0 && (
        <div className="space-y-6">
          {/* Completion Header */}
          <div className="text-center">
            <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <FileText className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">Analysis Complete!</h3>
            <p className="text-gray-600">
              Your invoice has been processed and analyzed by our AI agents
            </p>
          </div>

          {/* Quick Results Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {analysisResults.map((result, index) => (
              <div key={index} className="bg-white rounded-lg border-2 border-gray-100 p-4 text-center">
                <div className={`text-3xl font-bold mb-1 ${
                  result.overall_risk_score > 50 ? 'text-red-600' : 
                  result.overall_risk_score > 25 ? 'text-yellow-600' : 'text-green-600'
                }`}>
                  {result.overall_risk_score}%
                </div>
                <div className="text-sm text-gray-600 mb-2">Risk Score</div>
                <div className={`inline-flex px-3 py-1 rounded-full text-xs font-medium ${
                  result.status === 'approved' ? 'bg-green-100 text-green-800' :
                  result.status === 'rejected' ? 'bg-red-100 text-red-800' : 
                  'bg-yellow-100 text-yellow-800'
                }`}>
                  {(result.status || 'processing').replace('_', ' ').toUpperCase()}
                </div>
              </div>
            ))}
          </div>

          {/* Detailed Analysis Results */}
          {analysisResults.map((result, index) => (
            <InvoiceResponse
              key={index}
              invoiceData={uploads[index]}
              analysisResult={result}
            />
          ))}

          {/* Actions */}
          <div className="flex gap-3 justify-center pt-4">
            <button
              onClick={handleReset}
              className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors font-medium"
            >
              Process Another Invoice
            </button>
            <button 
              onClick={() => window.location.href = '#dashboard'}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              View Dashboard
            </button>
            <button className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium">
              Download Report
            </button>
          </div>
        </div>
      )}

      {/* Quick Stats */}
      {uploadState !== 'idle' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg border p-4 text-center">
            <Upload className="w-6 h-6 text-blue-500 mx-auto mb-2" />
            <div className="text-lg font-semibold text-gray-900">{uploads.length}</div>
            <div className="text-sm text-gray-600">Files Uploaded</div>
          </div>
          <div className="bg-white rounded-lg border p-4 text-center">
            <FileText className="w-6 h-6 text-green-500 mx-auto mb-2" />
            <div className="text-lg font-semibold text-gray-900">
              {uploads.filter(u => u.status === 'completed').length}
            </div>
            <div className="text-sm text-gray-600">Processed</div>
          </div>
          <div className="bg-white rounded-lg border p-4 text-center">
            <TrendingUp className="w-6 h-6 text-purple-500 mx-auto mb-2" />
            <div className="text-lg font-semibold text-gray-900">
              {analysisResults.length > 0 ? 
                Math.round(analysisResults[0]?.overall_risk_score || 0) : 0}%
            </div>
            <div className="text-sm text-gray-600">Avg Risk Score</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadInterface;