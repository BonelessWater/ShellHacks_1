import React, { useState } from 'react';
import Header from './common/Header';
import Navigation from './common/Navigation';
import Dashboard from './dashboard/Dashboard';
import UploadInterface from './upload/UploadInterface';
import Analytics from './analytics/Analytics';
import { useInvoices } from '../hooks/useInvoices';
import { useSystemStatus } from '../hooks/useSystemStatus';

const InvoiceVerificationSystem = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const { invoices, processing, handleFileUpload } = useInvoices();
  const { systemStatus } = useSystemStatus();

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard invoices={invoices} />;
      case 'upload':
        return <UploadInterface onFileUpload={handleFileUpload} processing={processing} />;
      case 'analytics':
        return <Analytics systemStatus={systemStatus} invoices={invoices} />;
      default:
        return <Dashboard invoices={invoices} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Header systemStatus={systemStatus} />
      <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {renderContent()}
      </main>
    </div>
  );
};

export default InvoiceVerificationSystem;