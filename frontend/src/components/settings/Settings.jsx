import React from 'react';
import AgentConfig from './AgentConfig';
import APIConfig from './APIConfig';

const Settings = () => {
  return (
    <div className="space-y-6">
      <AgentConfig />
      <APIConfig />
    </div>
  );
};

export default Settings;