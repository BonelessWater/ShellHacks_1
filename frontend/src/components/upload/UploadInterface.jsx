import React from 'react';
import FileDropzone from './FileDropzone';
import ProcessingStatus from './ProcessingStatus';

const UploadInterface = ({ onFileUpload, processing }) => {
  return (
    <div className="space-y-6">
      <FileDropzone onFileUpload={onFileUpload} processing={processing} />
      {processing && <ProcessingStatus />}
    </div>
  );
};

export default UploadInterface;