import React, { useState } from 'react';
import { BarChart3, PieChart, TrendingUp, Clock, DollarSign, AlertCircle, Calendar, Filter } from 'lucide-react';

const InvoiceAnalytics = ({ invoices }) => {
  const [timeFilter, setTimeFilter] = useState('all');
  const [amountFilter, setAmountFilter] = useState('all');

  // Filter invoices based on selected filters
  const filteredInvoices = invoices.filter(invoice => {
    const invoiceDate = new Date(invoice.date);
    const now = new Date();
    
    // Apply time filter
    let timeMatch = true;
    if (timeFilter === 'week') {
      timeMatch = (now - invoiceDate) / (1000 * 60 * 60 * 24) <= 7;
    } else if (timeFilter === 'month') {
      timeMatch = (now - invoiceDate) / (1000 * 60 * 60 * 24) <= 30;
    } else if (timeFilter === 'quarter') {
      timeMatch = (now - invoiceDate) / (1000 * 60 * 60 * 24) <= 90;
    }

    // Apply amount filter
    let amountMatch = true;
    if (amountFilter === 'small') {
      amountMatch = invoice.amount < 1000;
    } else if (amountFilter === 'medium') {
      amountMatch = invoice.amount >= 1000 && invoice.amount < 10000;
    } else if (amountFilter === 'large') {
      amountMatch = invoice.amount >= 10000;
    }

    return timeMatch && amountMatch;
  });

  // Calculate analytics
  const totalAmount = filteredInvoices.reduce((sum, inv) => sum + inv.amount, 0);
  const avgAmount = totalAmount / filteredInvoices.length || 0;
  const medianAmount = (() => {
    const sorted = [...filteredInvoices].sort((a, b) => a.amount - b.amount);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 
      ? (sorted[mid - 1].amount + sorted[mid].amount) / 2 
      : sorted[mid].amount;
  })();

  // Amount distribution
  const amountRanges = {
    'Under $1K': filteredInvoices.filter(inv => inv.amount < 1000).length,
    '$1K - $5K': filteredInvoices.filter(inv => inv.amount >= 1000 && inv.amount < 5000).length,
    '$5K - $10K': filteredInvoices.filter(inv => inv.amount >= 5000 && inv.amount < 10000).length,
    '$10K - $25K': filteredInvoices.filter(inv => inv.amount >= 10000 && inv.amount < 25000).length,
    'Over $25K': filteredInvoices.filter(inv => inv.amount >= 25000).length
  };

  // Status distribution
  const statusDistribution = {
    'Approved': filteredInvoices.filter(inv => inv.status === 'approved').length,
    'Review Required': filteredInvoices.filter(inv => inv.status === 'review_required').length,
    'Rejected': filteredInvoices.filter(inv => inv.status === 'rejected').length,
    'Processing': filteredInvoices.filter(inv => inv.status === 'processing').length
  };

  // Processing time analysis (simulated)
  const processingTimes = {
    'Under 1 minute': Math.floor(filteredInvoices.length * 0.65),
    '1-5 minutes': Math.floor(filteredInvoices.length * 0.25),
    '5-15 minutes': Math.floor(filteredInvoices.length * 0.08),
    'Over 15 minutes': Math.floor(filteredInvoices.length * 0.02)
  };

  // Issue frequency analysis
  const issueTypes = {
    'No Issues': filteredInvoices.filter(inv => inv.issues === 0).length,
    'Low Issues (1-2)': filteredInvoices.filter(inv => inv.issues >= 1 && inv.issues <= 2).length,
    'Medium Issues (3-4)': filteredInvoices.filter(inv => inv.issues >= 3 && inv.issues <= 4).length,
    'High Issues (5+)': filteredInvoices.filter(inv => inv.issues >= 5).length
  };

  // Top problematic invoices
  const problematicInvoices = [...filteredInvoices]
    .filter(inv => inv.issues > 0)
    .sort((a, b) => b.issues - a.issues)
    .slice(0, 5);

  // High value invoices
  const highValueInvoices = [...filteredInvoices]
    .sort((a, b) => b.amount - a.amount)
    .slice(0, 5);

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const getPercentage = (value, total) => {
    return total > 0 ? ((value / total) * 100).toFixed(1) : '0.0';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'approved': return 'text-green-600 bg-green-100';
      case 'rejected': return 'text-red-600 bg-red-100';
      case 'review_required': return 'text-yellow-600 bg-yellow-100';
      case 'processing': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center gap-4">
          <Filter className="w-5 h-5 text-gray-400" />
          <div className="flex gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Time Period</label>
              <select
                value={timeFilter}
                onChange={(e) => setTimeFilter(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="all">All Time</option>
                <option value="week">Last 7 Days</option>
                <option value="month">Last 30 Days</option>
                <option value="quarter">Last 90 Days</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Amount Range</label>
              <select
                value={amountFilter}
                onChange={(e) => setAmountFilter(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="all">All Amounts</option>
                <option value="small">Under $1K</option>
                <option value="medium">$1K - $10K</option>
                <option value="large">Over $10K</option>
              </select>
            </div>
          </div>
          <div className="ml-auto text-sm text-gray-600">
            Showing {filteredInvoices.length} of {invoices.length} invoices
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <DollarSign className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Average Amount</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(avgAmount)}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <TrendingUp className="h-8 w-8 text-blue-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Median Amount</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(medianAmount)}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Clock className="h-8 w-8 text-purple-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg Processing</p>
              <p className="text-2xl font-bold text-gray-900">1.2s</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <AlertCircle className="h-8 w-8 text-red-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Issue Rate</p>
              <p className="text-2xl font-bold text-gray-900">
                {getPercentage(filteredInvoices.filter(inv => inv.issues > 0).length, filteredInvoices.length)}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Distribution Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Amount Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <BarChart3 className="w-5 h-5 mr-2" />
            Amount Distribution
          </h3>
          <div className="space-y-3">
            {Object.entries(amountRanges).map(([range, count]) => (
              <div key={range} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{range}</span>
                <div className="flex items-center gap-3">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${(count / filteredInvoices.length) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-12">
                    {count} ({getPercentage(count, filteredInvoices.length)}%)
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Status Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <PieChart className="w-5 h-5 mr-2" />
            Status Distribution
          </h3>
          <div className="space-y-3">
            {Object.entries(statusDistribution).map(([status, count]) => (
              <div key={status} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{status}</span>
                <div className="flex items-center gap-3">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-indigo-600 h-2 rounded-full"
                      style={{ width: `${(count / filteredInvoices.length) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-12">
                    {count} ({getPercentage(count, filteredInvoices.length)}%)
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Processing Time and Issue Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Processing Time Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <Clock className="w-5 h-5 mr-2" />
            Processing Time Analysis
          </h3>
          <div className="space-y-3">
            {Object.entries(processingTimes).map(([timeRange, count]) => (
              <div key={timeRange} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{timeRange}</span>
                <div className="flex items-center gap-3">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-600 h-2 rounded-full"
                      style={{ width: `${(count / filteredInvoices.length) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-12">
                    {count} ({getPercentage(count, filteredInvoices.length)}%)
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Issue Frequency */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <AlertCircle className="w-5 h-5 mr-2" />
            Issue Frequency Analysis
          </h3>
          <div className="space-y-3">
            {Object.entries(issueTypes).map(([issueLevel, count]) => (
              <div key={issueLevel} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{issueLevel}</span>
                <div className="flex items-center gap-3">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-red-600 h-2 rounded-full"
                      style={{ width: `${(count / filteredInvoices.length) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-12">
                    {count} ({getPercentage(count, filteredInvoices.length)}%)
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Top Lists */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Most Problematic Invoices */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Most Problematic Invoices</h3>
          <div className="space-y-3">
            {problematicInvoices.length > 0 ? problematicInvoices.map((invoice, index) => (
              <div key={invoice.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{invoice.id}</p>
                  <p className="text-sm text-gray-600">{invoice.vendor}</p>
                </div>
                <div className="text-right">
                  <p className="font-medium text-gray-900">{formatCurrency(invoice.amount)}</p>
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(invoice.status)}`}>
                    {invoice.issues} issues
                  </span>
                </div>
              </div>
            )) : (
              <p className="text-gray-500 text-center py-4">No problematic invoices found</p>
            )}
          </div>
        </div>

        {/* Highest Value Invoices */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Highest Value Invoices</h3>
          <div className="space-y-3">
            {highValueInvoices.map((invoice, index) => (
              <div key={invoice.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{invoice.id}</p>
                  <p className="text-sm text-gray-600">{invoice.vendor}</p>
                </div>
                <div className="text-right">
                  <p className="font-medium text-gray-900">{formatCurrency(invoice.amount)}</p>
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(invoice.status)}`}>
                    {invoice.status.replace('_', ' ')}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Insights */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Key Invoice Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="space-y-2">
            <p className="font-medium text-gray-900">Volume Trends</p>
            <p className="text-sm text-gray-600">
              {filteredInvoices.length > invoices.length * 0.8 
                ? "High invoice volume period - ensure adequate processing capacity"
                : "Normal invoice volume - systems operating within capacity"}
            </p>
          </div>
          <div className="space-y-2">
            <p className="font-medium text-gray-900">Amount Patterns</p>
            <p className="text-sm text-gray-600">
              {avgAmount > 5000 
                ? "Higher than average invoice amounts detected - review for bulk purchasing opportunities"
                : "Invoice amounts within normal ranges"}
            </p>
          </div>
          <div className="space-y-2">
            <p className="font-medium text-gray-900">Quality Assessment</p>
            <p className="text-sm text-gray-600">
              {filteredInvoices.filter(inv => inv.issues === 0).length / filteredInvoices.length > 0.8
                ? "Excellent invoice quality - low error rates detected"
                : "Some invoices require attention - consider vendor training"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InvoiceAnalytics;