import React, { useState, useEffect } from 'react';
import SummaryCards from './SummaryCards';
import InvoiceTable from './InvoiceTable';
import InvoiceFilters from './InvoiceFilters';

const Dashboard = ({ invoices }) => {
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