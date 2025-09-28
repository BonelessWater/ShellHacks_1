// Enhanced Upload Interface with Real Backend Integration
// frontend/src/components/upload/UploadInterface.jsx
import React, { useState, useCallback } from 'react';
import FileDropzone from './FileDropzone';
import UploadProgress from './UploadProgress';
import InvoiceResponse from './InvoiceResponse';
import { mapBackendToUI } from '../utils/mapAnalysisResult';

const UploadInterface = ({ onFileUpload, processing }) => {
  const [uploads, setUploads] = useState([]);
  const [analysisResults, setAnalysisResults] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const simulateRealisticProgress = (uploadId, startTime) => {
    const duration = 60000; // 60 seconds total
    const interval = 500; // Update every 500ms
    
    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const rawProgress = Math.min((elapsed / duration) * 100, 99); // Cap at 99% until real completion
      
      // Create realistic progress curve (fast start, slower middle, final push)
      let adjustedProgress;
      if (rawProgress < 20) {
        adjustedProgress = rawProgress * 1.5; // Fast initial progress
      } else if (rawProgress < 70) {
        adjustedProgress = 20 + (rawProgress - 20) * 0.8; // Slower middle
      } else {
        adjustedProgress = 60 + (rawProgress - 70) * 1.3; // Final push
      }
      
      // Determine current step based on progress
      let currentStep;
      if (adjustedProgress < 15) {
        currentStep = 'Uploading file...';
      } else if (adjustedProgress < 30) {
        currentStep = 'Extracting invoice data...';
      } else if (adjustedProgress < 50) {
        currentStep = 'Running fraud detection agents...';
      } else if (adjustedProgress < 70) {
        currentStep = 'Analyzing vendor information...';
      } else if (adjustedProgress < 85) {
        currentStep = 'Validating amounts and duplicates...';
      } else if (adjustedProgress < 95) {
        currentStep = 'Cross-referencing databases...';
      } else {
        currentStep = 'Finalizing risk assessment...';
      }
      
      setUploads(prev => prev.map(upload => 
        upload.id === uploadId 
          ? { 
              ...upload, 
              progress: Math.min(adjustedProgress, 99),
              currentStep: currentStep,
              status: 'processing'
            }
          : upload
      ));
      
      // Clean up interval when we reach the end
      if (elapsed >= duration) {
        clearInterval(progressInterval);
      }
    }, interval);
    
    return progressInterval;
  };

  const handleFileUpload = useCallback(async (files) => {
    if (!files || files.length === 0) return;

    setIsProcessing(true);
    const startTime = Date.now();
    
    // Create upload tracking objects
    const newUploads = Array.from(files).map((file, index) => ({
      id: `upload_${Date.now()}_${index}`,
      fileName: file.name,
      status: 'processing',
      progress: 0,
      currentStep: 'Preparing upload...',
      file: file
    }));
    
    setUploads(newUploads);
    setAnalysisResults([]);

    // Process each file
    for (const upload of newUploads) {
      try {
        // Start progress simulation
        const progressInterval = simulateRealisticProgress(upload.id, startTime);
        
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('file', upload.file);
        
        // Make actual API call to backend
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });
        
        const result = await response.json();
        
        // Clear the progress interval
        clearInterval(progressInterval);
        
        if (result.success) {
          // Update upload status to completed
          setUploads(prev => prev.map(u => 
            u.id === upload.id 
              ? { 
                  ...u, 
                  status: 'completed',
                  progress: 100,
                  currentStep: 'Analysis complete!',
                  result: result.data,
                  processingTime: (Date.now() - startTime) / 1000
                }
              : u
          ));
          
          // Map backend response to UI format and add to analysis results
          const mappedResult = mapBackendToUI(result);
          if (mappedResult) {
            setAnalysisResults(prev => [...prev, {
              id: upload.id,
              fileName: upload.fileName,
              ...mappedResult
            }]);
          } else {
            console.error('Failed to map backend result:', result);
            // Fallback: try to use raw result data
            setAnalysisResults(prev => [...prev, {
              id: upload.id,
              fileName: upload.fileName,
              ...result.data
            }]);
          }
        } else {
          // Handle upload failure
          setUploads(prev => prev.map(u => 
            u.id === upload.id 
              ? { 
                  ...u, 
                  status: 'failed',
                  progress: 0,
                  currentStep: 'Failed',
                  error: result.message || 'Upload failed'
                }
              : u
          ));
        }
        
      } catch (error) {
        console.error('Upload error:', error);
        
        setUploads(prev => prev.map(u => 
          u.id === upload.id 
            ? { 
                ...u, 
                status: 'failed',
                progress: 0,
                currentStep: 'Failed',
                error: error.message || 'Network error'
              }
            : u
        ));
      }
    }
    
    setIsProcessing(false);
  }, []);

  const handleComplete = () => {
    setUploads([]);
    setAnalysisResults([]);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Invoice Fraud Detection
        </h1>
        <p className="text-gray-600">
          Upload your invoices for AI-powered fraud analysis
        </p>
      </div>

      {/* File Upload Area */}
      <FileDropzone 
        onFilesSelected={handleFileUpload}
        disabled={isProcessing}
      />

      {/* Progress Display */}
      {uploads.length > 0 && (
        <UploadProgress 
          uploads={uploads} 
          onComplete={handleComplete}
        />
      )}

      {/* Results Display */}
      {analysisResults.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold text-gray-900">
            Analysis Results
          </h2>
          {analysisResults.map((result) => (
            <InvoiceResponse 
              key={result.id}
              analysisResult={result}
              invoiceData={{ fileName: result.fileName }}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default UploadInterface;