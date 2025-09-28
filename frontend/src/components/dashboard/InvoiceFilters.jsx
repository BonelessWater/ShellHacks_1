import React, { useState } from 'react';
import { Search, Filter, X } from 'lucide-react';

const InvoiceFilters = ({ invoices, onFilterChange }) => {
  const [showFilters, setShowFilters] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState({
    confidenceLevel: 'all',
    issueCount: 'all',
    status: 'all',
    amountRange: 'all',
    riskLevel: 'all'
  });

  const handleSearchChange = (value) => {
    setSearchTerm(value);
    applyFilters({ ...filters }, value);
  };

  const handleFilterChange = (filterType, value) => {
    const newFilters = { ...filters, [filterType]: value };
    setFilters(newFilters);
    applyFilters(newFilters, searchTerm);
  };

  const clearAllFilters = () => {
    const clearedFilters = {
      confidenceLevel: 'all',
      issueCount: 'all',
      status: 'all',
      amountRange: 'all',
      riskLevel: 'all'
    };
    setFilters(clearedFilters);
    setSearchTerm('');
    applyFilters(clearedFilters, '');
  };

  const applyFilters = (currentFilters, currentSearchTerm) => {
    let filtered = [...invoices];

    // Apply search filter
    if (currentSearchTerm) {
      filtered = filtered.filter(invoice =>
        invoice.vendor.toLowerCase().includes(currentSearchTerm.toLowerCase()) ||
        invoice.id.toLowerCase().includes(currentSearchTerm.toLowerCase()) ||
        (invoice.description && invoice.description.toLowerCase().includes(currentSearchTerm.toLowerCase()))
      );
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

    onFilterChange(filtered);
  };

  const getActiveFilterCount = () => {
    const activeFilters = Object.values(filters).filter(value => value !== 'all').length;
    return searchTerm ? activeFilters + 1 : activeFilters;
  };

  return (
    <div className="bg-white shadow rounded-lg p-4 mb-6">
      {/* Search and Filter Toggle */}
      <div className="flex gap-4 items-center flex-wrap">
        <div className="flex-1 min-w-64">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search invoices by vendor, ID, or description..."
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
      </div>

      {/* Filter Options */}
      {showFilters && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {/* Confidence Level Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Confidence Level
              </label>
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

            {/* Issue Count Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Issues Found
              </label>
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

            {/* Status Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Status
              </label>
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

            {/* Amount Range Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Amount Range
              </label>
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

            {/* Risk Level Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Risk Level
              </label>
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
          </div>

          {/* Filter Actions */}
          <div className="flex justify-between items-center mt-4">
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
        </div>
      )}
    </div>
  );
};

export default InvoiceFilters;