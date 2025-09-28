// Enhanced FileDropzone Component
// frontend/src/components/upload/FileDropzone.jsx
import React, { useState, useRef } from 'react';
import { Upload, X, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { apiService } from '../../services/api';

const FileDropzone = ({ onFileUpload, processing }) => {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({});
  const [uploadResults, setUploadResults] = useState({});
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    handleFiles(files);
  };

  const handleFiles = (files) => {
    const validFiles = files.filter(file => {
      const validTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg', 
                         'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         'application/vnd.ms-excel', 'application/json'];
      return validTypes.includes(file.type) || file.name.toLowerCase().match(/\.(pdf|png|jpg|jpeg|xlsx|xls|json)$/);
    });

    setSelectedFiles(prev => [...prev, ...validFiles]);
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
    setUploadResults(prev => {
      const newResults = { ...prev };
      delete newResults[index];
      return newResults;
    });
  };

  const uploadFiles = async () => {
    if (selectedFiles.length === 0) return;

    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      setUploadProgress(prev => ({ ...prev, [i]: 0 }));

      try {
        // Simulate progress
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => ({
            ...prev,
            [i]: Math.min((prev[i] || 0) + Math.random() * 30, 90)
          }));
        }, 200);

        const result = await apiService.uploadFile(file);
        
        clearInterval(progressInterval);
        setUploadProgress(prev => ({ ...prev, [i]: 100 }));
        setUploadResults(prev => ({ ...prev, [i]: { success: true, data: result } }));

      } catch (error) {
        setUploadProgress(prev => ({ ...prev, [i]: 100 }));
        setUploadResults(prev => ({ ...prev, [i]: { success: false, error: error.message } }));
      }
    }

    // Call parent handler
    if (onFileUpload) {
      const mockEvent = { target: { files: selectedFiles } };
      onFileUpload(mockEvent);
    }
  };

  const clearAll = () => {
    setSelectedFiles([]);
    setUploadProgress({});
    setUploadResults({});
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
          dragOver 
            ? 'border-blue-400 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <Upload className={`mx-auto h-12 w-12 ${dragOver ? 'text-blue-500' : 'text-gray-400'}`} />
        <div className="mt-4">
          <p className="text-lg font-medium text-gray-900">
            {dragOver ? 'Drop files here' : 'Upload Invoice Files'}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            Drag and drop files here, or click to select
          </p>
          <p className="text-xs text-gray-400 mt-2">
            Supports: PDF, PNG, JPG, Excel, JSON (Max 10MB each)
          </p>
        </div>
        
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg,.xlsx,.xls,.json"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        <button
          onClick={() => fileInputRef.current?.click()}
          className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          disabled={processing}
        >
          Select Files
        </button>
      </div>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <div className="bg-white rounded-lg border p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium">Selected Files ({selectedFiles.length})</h3>
            <div className="space-x-2">
              <button
                onClick={uploadFiles}
                disabled={processing || Object.keys(uploadResults).length > 0}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400 transition-colors"
              >
                Upload All
              </button>
              <button
                onClick={clearAll}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
              >
                Clear All
              </button>
            </div>
          </div>

          <div className="space-y-3">
            {selectedFiles.map((file, index) => (
              <FileItem
                key={index}
                file={file}
                index={index}
                progress={uploadProgress[index]}
                result={uploadResults[index]}
                onRemove={removeFile}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// File Item Component
const FileItem = ({ file, index, progress, result, onRemove }) => {
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = () => {
    if (result?.success) return <CheckCircle className="w-5 h-5 text-green-500" />;
    if (result?.error) return <AlertCircle className="w-5 h-5 text-red-500" />;
    if (progress !== undefined) return <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
    return <FileText className="w-5 h-5 text-gray-400" />;
  };

  return (
    <div className="flex items-center gap-3 p-3 border rounded-lg">
      {getStatusIcon()}
      
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
        <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
        
        {progress !== undefined && progress < 100 && (
          <div className="mt-2 w-full bg-gray-200 rounded-full h-1">
            <div 
              className="bg-blue-600 h-1 rounded-full transition-all duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
        
        {result?.error && (
          <p className="text-xs text-red-600 mt-1">{result.error}</p>
        )}
        
        {result?.success && (
          <p className="text-xs text-green-600 mt-1">Upload successful</p>
        )}
      </div>

      {!result && (
        <button
          onClick={() => onRemove(index)}
          className="p-1 text-gray-400 hover:text-red-500 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

export default FileDropzone;