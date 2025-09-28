import React, { useState, useEffect } from 'react';
import { Search, Filter, X, Download, SortAsc, SortDesc } from 'lucide-react';

const InvoiceFilters = ({ invoices, onFilterChange, onSortChange, onExport, sortConfig: propSortConfig }) => {
  const [showFilters, setShowFilters] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState(propSortConfig || { field: 'date', direction: 'desc' });
  const [showExportMenu, setShowExportMenu] = useState(false);
  
  const [filters, setFilters] = useState({
    confidenceLevel: 'all',
    issueCount: 'all',
    status: 'all',
    amountRange: 'all',
    riskLevel: 'all',
    dateRange: 'all'
  });

  // Sync internal sortConfig with prop changes
  useEffect(() => {
    if (propSortConfig) {
      setSortConfig(propSortConfig);
    }
  }, [propSortConfig]);

  const handleSearchChange = (value) => {
    setSearchTerm(value);
    applyFilters({ ...filters }, value);
  };

  const handleFilterChange = (filterType, value) => {
    const newFilters = { ...filters, [filterType]: value };
    setFilters(newFilters);
    applyFilters(newFilters, searchTerm);
  };

  const handleSort = (field) => {
    const direction = sortConfig.field === field && sortConfig.direction === 'desc' ? 'asc' : 'desc';
    const newSortConfig = { field, direction };
    setSortConfig(newSortConfig);
    if (onSortChange) {
      onSortChange(newSortConfig);
    }
  };

  const clearAllFilters = () => {
    const clearedFilters = {
      confidenceLevel: 'all',
      issueCount: 'all',
      status: 'all',
      amountRange: 'all',
      riskLevel: 'all',
      dateRange: 'all'
    };
    setFilters(clearedFilters);
    setSearchTerm('');
    applyFilters(clearedFilters, '');
  };

  const applyFilters = (currentFilters, currentSearchTerm) => {
    let filtered = [...invoices];

    // Apply search filter
    if (currentSearchTerm) {
      const term = currentSearchTerm.toLowerCase();
      filtered = filtered.filter(invoice => {
        const vendor = (typeof invoice.vendor === 'string') ? invoice.vendor : (invoice.vendor && invoice.vendor.name) ? invoice.vendor.name : '';
        return (
          vendor.toLowerCase().includes(term) ||
          (invoice.id && invoice.id.toLowerCase().includes(term)) ||
          (invoice.description && invoice.description.toLowerCase().includes(term))
        );
      });
    }

    // Apply confidence level filter
    if (currentFilters.confidenceLevel !== 'all') {
      filtered = filtered.filter(invoice => {
        const confidence = invoice.confidence * 100;
        switch (currentFilters.confidenceLevel) {
          case 'high':
            return confidence >= 90;
          case 'medium':
            return confidence >= 70 && confidence < 90;
          case 'low':
            return confidence < 70;
          default:
            return true;
        }
      });
    }

    // Apply issue count filter
    if (currentFilters.issueCount !== 'all') {
      filtered = filtered.filter(invoice => {
        switch (currentFilters.issueCount) {
          case 'none':
            return invoice.issues === 0;
          case 'low':
            return invoice.issues >= 1 && invoice.issues <= 2;
          case 'high':
            return invoice.issues > 2;
          default:
            return true;
        }
      });
    }

    // Apply status filter
    if (currentFilters.status !== 'all') {
      filtered = filtered.filter(invoice => invoice.status === currentFilters.status);
    }

    // Apply amount range filter
    if (currentFilters.amountRange !== 'all') {
      filtered = filtered.filter(invoice => {
        switch (currentFilters.amountRange) {
          case 'small':
            return invoice.amount < 1000;
          case 'medium':
            return invoice.amount >= 1000 && invoice.amount < 10000;
          case 'large':
            return invoice.amount >= 10000;
          default:
            return true;
        }
      });
    }

    // Apply risk level filter
    if (currentFilters.riskLevel !== 'all') {
      filtered = filtered.filter(invoice => {
        const riskLevel = invoice.issues > 2 ? 'high' : invoice.issues > 0 ? 'medium' : 'low';
        return riskLevel === currentFilters.riskLevel;
      });
    }

    // Apply date range filter
    if (currentFilters.dateRange !== 'all') {
      const now = new Date();
      filtered = filtered.filter(invoice => {
        const invoiceDate = new Date(invoice.date);
        switch (currentFilters.dateRange) {
          case 'today':
            return invoiceDate.toDateString() === now.toDateString();
          case 'week':
            return (now - invoiceDate) / (1000 * 60 * 60 * 24) <= 7;
          case 'month':
            return (now - invoiceDate) / (1000 * 60 * 60 * 24) <= 30;
          case 'quarter':
            return (now - invoiceDate) / (1000 * 60 * 60 * 24) <= 90;
          default: 
            return true;
        }
      });
    }

    onFilterChange(filtered);
  };

  const getActiveFilterCount = () => {
    const activeFilters = Object.values(filters).filter(value => value !== 'all').length;
    return searchTerm ? activeFilters + 1 : activeFilters;
  };

  const exportData = (format) => {
    if (onExport) {
      onExport(format);
    }
    setShowExportMenu(false);
  };

  return (
    <div className="bg-white shadow rounded-lg p-4 mb-6">
      {/* Main Search and Controls */}
      <div className="flex gap-4 items-center flex-wrap mb-4">
        <div className="flex-1 min-w-64">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search invoices, vendors, descriptions..."
              value={searchTerm}
              onChange={(e) => handleSearchChange(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            />
            {searchTerm && (
              <button
                onClick={() => handleSearchChange('')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
        
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`flex items-center gap-2 px-4 py-2 border rounded-md transition-colors ${
            showFilters 
              ? 'bg-indigo-50 border-indigo-300 text-indigo-700' 
              : 'border-gray-300 text-gray-700 hover:bg-gray-50'
          }`}
        >
          <Filter className="w-4 h-4" />
          Filters
          {getActiveFilterCount() > 0 && (
            <span className="bg-indigo-100 text-indigo-800 text-xs font-medium px-2 py-0.5 rounded-full">
              {getActiveFilterCount()}
            </span>
          )}
        </button>

        <div className="relative">
          <button 
            onClick={() => setShowExportMenu(!showExportMenu)}
            className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
          {showExportMenu && (
            <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-200 z-10">
              <div className="py-1">
                <button
                  onClick={() => exportData('csv')}
                  className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  Export as CSV
                </button>
                <button
                  onClick={() => exportData('json')}
                  className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  Export as JSON
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Sort Controls */}
      <div className="flex gap-2 mb-4 text-sm">
        <span className="text-gray-600">Sort by:</span>
        {['date', 'amount', 'vendor', 'confidence', 'risk'].map(field => (
          <button
            key={field}
            onClick={() => handleSort(field)}
            className={`flex items-center gap-1 px-2 py-1 rounded transition-colors ${
              sortConfig.field === field 
                ? 'bg-indigo-100 text-indigo-700' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {field.charAt(0).toUpperCase() + field.slice(1)}
            {sortConfig.field === field && (
              sortConfig.direction === 'desc' ? 
                <SortDesc className="w-3 h-3" /> : 
                <SortAsc className="w-3 h-3" />
            )}
          </button>
        ))}
      </div>

      {/* Filter Options */}
      {showFilters && (
        <div className="border-t border-gray-200 pt-4">
          <div className="grid grid-cols-1 md:grid-cols-6 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
              <select
                value={filters.status}
                onChange={(e) => handleFilterChange('status', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              >
                <option value="all">All Status</option>
                <option value="approved">Approved</option>
                <option value="review_required">Review Required</option>
                <option value="rejected">Rejected</option>
                <option value="processing">Processing</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Risk Level</label>
              <select
                value={filters.riskLevel}
                onChange={(e) => handleFilterChange('riskLevel', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              >
                <option value="all">All Risk Levels</option>
                <option value="low">Low Risk</option>
                <option value="medium">Medium Risk</option>
                <option value="high">High Risk</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Amount Range</label>
              <select
                value={filters.amountRange}
                onChange={(e) => handleFilterChange('amountRange', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              >
                <option value="all">All Amounts</option>
                <option value="small">Small (&lt;$1K)</option>
                <option value="medium">Medium ($1K-$10K)</option>
                <option value="large">Large ($10K+)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confidence</label>
              <select
                value={filters.confidenceLevel}
                onChange={(e) => handleFilterChange('confidenceLevel', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              >
                <option value="all">All Levels</option>
                <option value="high">High (90%+)</option>
                <option value="medium">Medium (70-89%)</option>
                <option value="low">Low (&lt;70%)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Issues Found</label>
              <select
                value={filters.issueCount}
                onChange={(e) => handleFilterChange('issueCount', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              >
                <option value="all">All Counts</option>
                <option value="none">No Issues (0)</option>
                <option value="low">Low Issues (1-2)</option>
                <option value="high">High Issues (3+)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date Range</label>
              <select
                value={filters.dateRange}
                onChange={(e) => handleFilterChange('dateRange', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              >
                <option value="all">All Dates</option>
                <option value="today">Today</option>
                <option value="week">Last 7 Days</option>
                <option value="month">Last 30 Days</option>
                <option value="quarter">Last 90 Days</option>
              </select>
            </div>
          </div>

          {/* Filter Actions */}
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-600">
              {getActiveFilterCount() > 0 && (
                <span>
                  {getActiveFilterCount()} filter{getActiveFilterCount() !== 1 ? 's' : ''} applied
                </span>
              )}
            </div>
            <button
              onClick={clearAllFilters}
              className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors"
            >
              Clear All Filters
            </button>
          </div>

          {/* Active Filters Summary */}
          {getActiveFilterCount() > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="text-sm text-gray-600 mb-2">
                Active filters ({getActiveFilterCount()}):
              </div>
              <div className="flex flex-wrap gap-2">
                {searchTerm && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800">
                    Search: "{searchTerm}"
                    <button onClick={() => handleSearchChange('')} className="ml-1 text-blue-600 hover:text-blue-800">
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                )}
                {Object.entries(filters).map(([key, value]) => {
                  if (value !== 'all') {
                    return (
                      <span key={key} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-800">
                        {key}: {value}
                        <button onClick={() => handleFilterChange(key, 'all')} className="ml-1 text-gray-600 hover:text-gray-800">
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    );
                  }
                  return null;
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default InvoiceFilters;