// frontend/src/App.jsx
import React, { useState } from 'react';
import Header from './components/common/Header';
import Navigation from './components/common/Navigation';
import Dashboard from './components/dashboard/Dashboard';
import UploadInterface from './components/upload/UploadInterface';
import Analytics from './components/analytics/Analytics';
import Settings from './components/settings/Settings';
import { useInvoices } from './hooks/useInvoices';
import { useSystemStatus } from './hooks/useSystemStatus';

function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const { invoices, processing, error, backendConnected, handleFileUpload } = useInvoices();
  const { systemStatus, backendConnected: systemBackendConnected } = useSystemStatus();

  const renderContent = () => {
    switch (activeView) {
      case 'dashboard':
        return (
          <Dashboard 
            invoices={invoices} 
            backendConnected={backendConnected}
            error={error}
          />
        );
      case 'upload':
        return (
          <UploadInterface 
            onFileUpload={handleFileUpload}
            processing={processing}
            error={error}
          />
        );
      case 'analytics':
        return <Analytics invoices={invoices} systemStatus={systemStatus} />;
      case 'settings':
        return <Settings backendConnected={backendConnected} />;
      default:
        return (
          <Dashboard 
            invoices={invoices} 
            backendConnected={backendConnected}
            error={error}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        activeView={activeView}
        onViewChange={setActiveView}
        systemStatus={systemStatus}
        backendConnected={backendConnected || systemBackendConnected}
      />
      
      <Navigation 
        activeTab={activeView}
        onTabChange={setActiveView}
      />
      
      <main className="p-6">
        {renderContent()}
      </main>
    </div>
  );
}

export default App;