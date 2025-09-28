// frontend/src/components/upload/FileDropzone.jsx
import React, { useState } from 'react';
import { Upload } from 'lucide-react';

const FileDropzone = ({ onFileUpload, processing }) => {
  const [message, setMessage] = useState('');

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (response.ok) {
        setMessage(`✅ ${result.message}`);
      } else {
        setMessage(`❌ Upload failed: ${result.detail}`);
      }
    } catch (error) {
      setMessage(`❌ Error: ${error.message}`);
    }

    // Call parent callback if provided
    if (onFileUpload) {
      onFileUpload(event);
    }
  };

  return (
    <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors">
      <Upload className="mx-auto h-12 w-12 text-gray-400" />
      <div className="mt-4">
        <label htmlFor="file-upload" className="cursor-pointer">
          <span className="mt-2 block text-sm font-medium text-gray-900">
            Upload invoice files
          </span>
          <span className="mt-1 block text-sm text-gray-500">
            Supports PDF, PNG, JPG, and Excel files
          </span>
        </label>
        <input
          id="file-upload"
          name="file-upload"
          type="file"
          accept=".pdf,.png,.jpg,.jpeg,.xlsx,.xls"
          className="sr-only"
          onChange={handleFileChange}
        />
      </div>
      <button
        onClick={() => document.getElementById('file-upload').click()}
        className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        disabled={processing}
      >
        {processing ? 'Processing...' : 'Select Files'}
      </button>

      {/* Success/Error Message */}
      {message && (
        <div className={`mt-4 p-3 rounded-md ${
          message.includes('✅') 
            ? 'bg-green-50 border border-green-200 text-green-800' 
            : 'bg-red-50 border border-red-200 text-red-800'
        }`}>
          {message}
        </div>
      )}
    </div>
  );
};

export default FileDropzone;