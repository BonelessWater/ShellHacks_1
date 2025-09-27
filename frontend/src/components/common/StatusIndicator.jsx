import React from 'react';

const StatusIndicator = ({ isOnline, text }) => {
  return (
    <div className="flex items-center text-sm text-gray-500">
      <div className={`h-2 w-2 rounded-full mr-2 ${
        isOnline ? 'bg-green-400' : 'bg-red-400'
      }`}></div>
      {text}
    </div>
  );
};

export default StatusIndicator;