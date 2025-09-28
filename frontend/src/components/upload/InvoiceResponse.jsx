// Invoice Response Display Component
// frontend/src/components/upload/InvoiceResponse.jsx
import React, { useState } from 'react';
import { CheckCircle, AlertTriangle, XCircle, Eye, Download, Clock, Shield, FileText } from 'lucide-react';

const InvoiceResponse = ({ invoiceData, analysisResult }) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!analysisResult) {
    return (
      <div className="bg-gray-50 rounded-lg p-6 text-center">
        <Clock className="w-8 h-8 text-gray-400 mx-auto mb-2" />
        <p className="text-gray-600">Waiting for analysis results...</p>
      </div>
    );
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'approved': return 'green';
      case 'rejected': return 'red';
      case 'review_required': return 'yellow';
      default: return 'gray';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'approved': return <CheckCircle className="w-6 h-6 text-green-500" />;
      case 'rejected': return <XCircle className="w-6 h-6 text-red-500" />;
      case 'review_required': return <AlertTriangle className="w-6 h-6 text-yellow-500" />;
      default: return <Clock className="w-6 h-6 text-gray-500" />;
    }
  };

  const getRiskColor = (score) => {
    if (score >= 70) return 'red';
    if (score >= 40) return 'yellow';
    return 'green';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg border">
      {/* Header */}
      <div className="p-6 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {getStatusIcon(analysisResult.status)}
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                Invoice Analysis Complete
              </h3>
              <p className="text-sm text-gray-600">
                Invoice ID: {analysisResult.invoice_id}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className={`inline-flex px-3 py-1 rounded-full text-sm font-medium bg-${getStatusColor(analysisResult.status)}-100 text-${getStatusColor(analysisResult.status)}-800`}>
              {analysisResult.status.replace('_', ' ').toUpperCase()}
            </div>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="p-6 grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className={`text-2xl font-bold text-${getRiskColor(analysisResult.overall_risk_score)}-600`}>
            {Math.round(analysisResult.overall_risk_score)}%
          </div>
          <div className="text-sm text-gray-600">Risk Score</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">
            {Math.round(analysisResult.confidence * 100)}%
          </div>
          <div className="text-sm text-gray-600">Confidence</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900">
            {analysisResult.red_flags?.length || 0}
          </div>
          <div className="text-sm text-gray-600">Red Flags</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">
            {analysisResult.processing_time?.toFixed(1) || 'N/A'}s
          </div>
          <div className="text-sm text-gray-600">Processing Time</div>
        </div>
      </div>

      {/* Recommendation */}
      <div className="px-6 pb-4">
        <div className={`p-4 rounded-lg bg-${getStatusColor(analysisResult.status)}-50 border border-${getStatusColor(analysisResult.status)}-200`}>
          <h4 className={`font-medium text-${getStatusColor(analysisResult.status)}-800 mb-1`}>
            Recommendation
          </h4>
          <p className={`text-sm text-${getStatusColor(analysisResult.status)}-700`}>
            {analysisResult.recommendation}
          </p>
        </div>
      </div>

      {/* Red Flags */}
      {analysisResult.red_flags && analysisResult.red_flags.length > 0 && (
        <div className="px-6 pb-4">
          <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-500" />
            Detected Issues
          </h4>
          <div className="space-y-2">
            {analysisResult.red_flags.map((flag, index) => (
              <div key={index} className="flex items-start gap-2 p-2 bg-red-50 rounded border-l-2 border-red-400">
                <div className="w-2 h-2 bg-red-400 rounded-full mt-2 flex-shrink-0"></div>
                <span className="text-sm text-red-800">{flag}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Agent Results Summary */}
      {analysisResult.agent_results && analysisResult.agent_results.length > 0 && (
        <div className="px-6 pb-4">
          <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
            <Shield className="w-4 h-4 text-blue-500" />
            Agent Analysis ({analysisResult.agent_results.length} agents)
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {analysisResult.agent_results.slice(0, 4).map((agent, index) => (
              <div key={index} className="p-3 border rounded-lg bg-gray-50">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-900">
                    {agent.agent_id?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) || `Agent ${index + 1}`}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    agent.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {agent.status}
                  </span>
                </div>
                {agent.risk_score !== undefined && (
                  <div className="mt-2">
                    <div className="flex justify-between text-xs text-gray-600">
                      <span>Risk Level</span>
                      <span>{agent.risk_score}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                      <div 
                        className={`h-1 rounded-full ${
                          agent.risk_score > 70 ? 'bg-red-400' : 
                          agent.risk_score > 40 ? 'bg-yellow-400' : 'bg-green-400'
                        }`}
                        style={{ width: `${agent.risk_score}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="px-6 py-4 border-t bg-gray-50 flex items-center justify-between">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center gap-2 text-blue-600 hover:text-blue-700 transition-colors"
        >
          <Eye className="w-4 h-4" />
          {showDetails ? 'Hide Details' : 'View Details'}
        </button>
        
        <div className="flex gap-2">
          <button className="flex items-center gap-2 px-4 py-2 text-gray-700 bg-white border rounded-md hover:bg-gray-50 transition-colors">
            <Download className="w-4 h-4" />
            Export Report
          </button>
          <button className="flex items-center gap-2 px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors">
            <FileText className="w-4 h-4" />
            Process Next
          </button>
        </div>
      </div>

      {/* Detailed Analysis */}
      {showDetails && (
        <div className="border-t bg-gray-50">
          <div className="p-6 space-y-4">
            <h4 className="font-medium text-gray-900">Detailed Analysis</h4>
            
            {/* Analysis Summary */}
            {analysisResult.analysis_summary && (
              <div className="p-4 bg-white rounded border">
                <h5 className="font-medium text-gray-800 mb-2">Executive Summary</h5>
                <p className="text-sm text-gray-700">{analysisResult.analysis_summary}</p>
              </div>
            )}

            {/* All Agent Results */}
            {analysisResult.agent_results && (
              <div className="space-y-3">
                <h5 className="font-medium text-gray-800">Complete Agent Analysis</h5>
                {analysisResult.agent_results.map((agent, index) => (
                  <div key={index} className="p-4 bg-white rounded border">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-gray-900">
                        {agent.agent_id?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) || `Agent ${index + 1}`}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        agent.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {agent.status}
                      </span>
                    </div>
                    {agent.analysis && (
                      <p className="text-sm text-gray-700 mb-2">{agent.analysis}</p>
                    )}
                    {agent.findings && agent.findings.length > 0 && (
                      <div className="text-xs text-gray-600">
                        <strong>Findings:</strong> {agent.findings.join(', ')}
                      </div>
                    )}
                    {agent.execution_time && (
                      <div className="text-xs text-gray-500 mt-1">
                        Execution time: {agent.execution_time.toFixed(2)}s
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Metadata */}
            {analysisResult.metadata && (
              <div className="p-4 bg-white rounded border">
                <h5 className="font-medium text-gray-800 mb-2">Processing Metadata</h5>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Total Agents:</span>
                    <span className="ml-2 font-medium">{analysisResult.metadata.total_agents_executed}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Successful:</span>
                    <span className="ml-2 font-medium text-green-600">{analysisResult.metadata.successful_agents}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Failed:</span>
                    <span className="ml-2 font-medium text-red-600">{analysisResult.metadata.failed_agents}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Analysis Time:</span>
                    <span className="ml-2 font-medium">{analysisResult.metadata.analysis_timestamp}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default InvoiceResponse;