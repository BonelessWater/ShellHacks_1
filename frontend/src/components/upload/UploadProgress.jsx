// Enhanced Upload Progress Component with 60-second timeline
// frontend/src/components/upload/UploadProgress.jsx
import React from 'react';
import { Activity, CheckCircle, XCircle, Clock, Shield, FileSearch, TrendingUp, AlertTriangle } from 'lucide-react';

const UploadProgress = ({ uploads, onComplete }) => {
  if (!uploads || uploads.length === 0) return null;

  const totalFiles = uploads.length;
  const completedFiles = uploads.filter(u => u.status === 'completed' || u.status === 'failed').length;
  const overallProgress = (completedFiles / totalFiles) * 100;
  const isProcessing = uploads.some(u => u.status === 'processing');
  const currentProgress = uploads[0]?.progress || 0;

  const processingSteps = [
    { id: 'upload', label: 'Upload', icon: Clock, description: 'Transferring file securely', minProgress: 0 },
    { id: 'extraction', label: 'Extract Data', icon: FileSearch, description: 'Reading invoice content', minProgress: 15 },
    { id: 'detection', label: 'Fraud Detection', icon: Shield, description: 'Running AI analysis', minProgress: 30 },
    { id: 'validation', label: 'Validation', icon: AlertTriangle, description: 'Cross-checking data', minProgress: 60 },
    { id: 'analysis', label: 'Risk Analysis', icon: TrendingUp, description: 'Final assessment', minProgress: 85 },
    { id: 'completion', label: 'Complete', icon: CheckCircle, description: 'Processing finished', minProgress: 100 }
  ];

  const getCurrentStep = () => {
    for (let i = processingSteps.length - 1; i >= 0; i--) {
      if (currentProgress >= processingSteps[i].minProgress) {
        return i;
      }
    }
    return 0;
  };

  const currentStepIndex = getCurrentStep();

  return (
    <div className="bg-white rounded-lg shadow-lg border p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Processing Invoice</h3>
          <span className="text-sm text-gray-600">
            {isProcessing
            ? `${currentProgress.toFixed(2)}% complete`
            : `${completedFiles}/${totalFiles} completed`}
          </span>
        </div>
        
        {/* Overall Progress Bar */}
        <div className="mt-3">
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className={`h-3 rounded-full transition-all duration-500 ${
                isProcessing ? 'bg-blue-600' : 'bg-green-600'
              }`}
              style={{ width: `${isProcessing ? currentProgress : overallProgress}%` }}
            />
          </div>
          
          {/* Progress Milestones */}
          <div className="flex justify-between mt-2 text-xs text-gray-500">
            <span>0s</span>
            <span>15s</span>
            <span>30s</span>
            <span>45s</span>
            <span>60s</span>
          </div>
        </div>
      </div>

      {/* Processing Pipeline */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-gray-700 mb-4">AI Processing Pipeline</h4>
        <div className="space-y-3">
          {processingSteps.map((step, index) => {
            const Icon = step.icon;
            const isCompleted = currentProgress >= step.minProgress;
            const isActive = index === currentStepIndex && isProcessing;
            
            return (
              <div key={step.id} className={`flex items-center gap-4 p-3 rounded-lg transition-all duration-300 ${
                isActive ? 'bg-blue-50 border border-blue-200' :
                isCompleted ? 'bg-green-50 border border-green-200' : 
                'bg-gray-50 border border-gray-200'
              }`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 ${
                  isCompleted ? 'bg-green-500 text-white' :
                  isActive ? 'bg-blue-500 text-white' : 'bg-gray-300 text-gray-500'
                }`}>
                  {isActive ? (
                    <Activity className="w-4 h-4 animate-spin" />
                  ) : (
                    <Icon className="w-4 h-4" />
                  )}
                </div>
                
                <div className="flex-1">
                  <div className={`font-medium text-sm ${
                    isActive ? 'text-blue-900' :
                    isCompleted ? 'text-green-900' : 'text-gray-500'
                  }`}>
                    {step.label}
                  </div>
                  <div className={`text-xs ${
                    isActive ? 'text-blue-700' :
                    isCompleted ? 'text-green-700' : 'text-gray-400'
                  }`}>
                    {step.description}
                  </div>
                </div>
                
                {isCompleted && (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                )}
                
                {isActive && (
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-blue-600 font-medium">Running</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Individual File Progress */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-gray-700">File Status</h4>
        {uploads.map((upload, index) => (
          <FileProgress key={index} upload={upload} />
        ))}
      </div>

      {/* Time Estimation */}
      {isProcessing && (
        <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-800">
              {currentProgress < 90 
                ? `Estimated time remaining: ${Math.max(0, Math.round((100 - currentProgress) * 0.6))} seconds`
                : 'Finalizing analysis...'
              }
            </span>
          </div>
        </div>
      )}

      {/* Summary Stats */}
      {completedFiles === totalFiles && (
        <div className="mt-6 pt-4 border-t">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-lg font-semibold text-green-600">
                {uploads.filter(u => u.status === 'completed').length}
              </div>
              <div className="text-xs text-gray-600">Successful</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-red-600">
                {uploads.filter(u => u.status === 'failed').length}
              </div>
              <div className="text-xs text-gray-600">Failed</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-blue-600">
                {uploads.reduce((acc, u) => acc + (u.processingTime || 0), 0).toFixed(1)}s
              </div>
              <div className="text-xs text-gray-600">Total Time</div>
            </div>
          </div>
          
          {onComplete && (
            <button
              onClick={onComplete}
              className="w-full mt-4 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            >
              View Analysis Results
            </button>
          )}
        </div>
      )}
    </div>
  );
};

// Individual File Progress Component
const FileProgress = ({ upload }) => {
  const getStatusIcon = () => {
    switch (upload.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'processing':
        return <Activity className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (upload.status) {
      case 'completed': return 'green';
      case 'failed': return 'red';
      case 'processing': return 'blue';
      default: return 'gray';
    }
  };

  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border">
      {getStatusIcon()}
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-900 truncate">
            {upload.fileName}
          </span>
          <span className={`text-xs px-2 py-1 rounded-full bg-${getStatusColor()}-100 text-${getStatusColor()}-800`}>
            {upload.status}
          </span>
        </div>
        
        {upload.status === 'processing' && (
          <div className="mt-2">
            <div className="flex justify-between text-xs text-gray-600 mb-1">
              <span>{upload.currentStep || 'Processing...'}</span>
              <span>{upload.progress || 0}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1">
              <div 
                className="bg-blue-600 h-1 rounded-full transition-all duration-300"
                style={{ width: `${upload.progress || 0}%` }}
              />
            </div>
          </div>
        )}
        
        {upload.status === 'failed' && upload.error && (
          <p className="text-xs text-red-600 mt-1">{upload.error}</p>
        )}
        
        {upload.status === 'completed' && upload.result && (
          <div className="mt-1 text-xs text-gray-600">
            Risk Score: {upload.result.riskScore || 'N/A'}% • 
            Confidence: {upload.result.confidence || 'N/A'}%
          </div>
        )}
      </div>
      
      {upload.processingTime && (
        <div className="text-xs text-gray-500">
          {upload.processingTime.toFixed(1)}s
        </div>
      )}
    </div>
  );
};

export default UploadProgress;