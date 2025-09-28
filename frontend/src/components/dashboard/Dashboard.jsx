import React, { useState, useEffect } from 'react';
import SummaryCards from './SummaryCards';
import InvoiceTable from './InvoiceTable';
import InvoiceFilters from './InvoiceFilters';

const Dashboard = ({ invoices, backendConnected, error }) => {
  const [filteredInvoices, setFilteredInvoices] = useState(invoices);
  const [sortConfig, setSortConfig] = useState({ field: 'date', direction: 'desc' });

  // Update filtered invoices when invoices prop changes
  useEffect(() => {
    setFilteredInvoices(invoices);
  }, [invoices]);

  const handleFilterChange = (filtered) => {
    // Apply current sorting to the newly filtered data
    const sorted = applySorting(filtered, sortConfig);
    setFilteredInvoices(sorted);
  };

  const applySorting = (data, config) => {
    return [...data].sort((a, b) => {
      let aValue = a[config.field];
      let bValue = b[config.field];
      
      // Handle different data types
      switch (config.field) {
        case 'amount':
          aValue = parseFloat(aValue);
          bValue = parseFloat(bValue);
          break;
        case 'confidence':
          aValue = a.confidence * 100;
          bValue = b.confidence * 100;
          break;
        case 'risk':
          const riskOrder = { low: 1, medium: 2, high: 3 };
          aValue = riskOrder[a.issues > 2 ? 'high' : a.issues > 0 ? 'medium' : 'low'];
          bValue = riskOrder[b.issues > 2 ? 'high' : b.issues > 0 ? 'medium' : 'low'];
          break;
        case 'date':
          aValue = new Date(aValue);
          bValue = new Date(bValue);
          break;
        case 'vendor':
          aValue = String(aValue).toLowerCase();
          bValue = String(bValue).toLowerCase();
          break;
        default:
          aValue = String(aValue).toLowerCase();
          bValue = String(bValue).toLowerCase();
      }
      
      if (config.direction === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
  };

  const handleSortChange = (newSortConfig) => {
    setSortConfig(newSortConfig);
    // Apply sorting to current filtered invoices
    const sorted = applySorting(filteredInvoices, newSortConfig);
    setFilteredInvoices(sorted);
  };

  const handleExport = (format) => {
    // Simple export functionality
    const dataToExport = filteredInvoices.map(invoice => ({
      ID: invoice.id,
      Vendor: invoice.vendor,
      Amount: invoice.amount,
      Status: invoice.status,
      Confidence: Math.round(invoice.confidence * 100) + '%',
      Issues: invoice.issues,
      Date: invoice.date,
      Description: invoice.description || ''
    }));

    if (format === 'csv') {
      const csv = [
        Object.keys(dataToExport[0]).join(','),
        ...dataToExport.map(row => Object.values(row).map(value => 
          typeof value === 'string' && value.includes(',') ? `"${value}"` : value
        ).join(','))
      ].join('\n');
      
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `invoices_${new Date().toISOString().split('T')[0]}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    } else if (format === 'json') {
      const json = JSON.stringify(dataToExport, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `invoices_${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      window.URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="space-y-6">
      {/* Connection Status Banner */}
      {error && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">
                Backend Connection Issue
              </h3>
              <div className="mt-2 text-sm text-yellow-700">
                <p>{error} Showing cached/demo data.</p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {backendConnected && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-green-800">
                Backend Connected
              </h3>
              <div className="mt-2 text-sm text-green-700">
                <p>Successfully connected to invoice processing backend.</p>
              </div>
            </div>
          </div>
        </div>
      )}

      <SummaryCards invoices={filteredInvoices} />
      <InvoiceFilters 
        invoices={invoices} 
        onFilterChange={handleFilterChange}
        onSortChange={handleSortChange}
        onExport={handleExport}
        sortConfig={sortConfig}
      />
      <InvoiceTable 
        invoices={invoices} 
        filteredInvoices={filteredInvoices} 
      />
    </div>
  );
};

export default Dashboard;