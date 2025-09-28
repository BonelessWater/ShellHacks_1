import React, { useState } from 'react';
import { Users, TrendingUp, AlertTriangle, DollarSign, Clock, Star, Filter, Eye } from 'lucide-react';

const VendorAnalytics = ({ invoices }) => {
  const [sortBy, setSortBy] = useState('totalValue');
  const [showRiskOnly, setShowRiskOnly] = useState(false);
  const [selectedVendor, setSelectedVendor] = useState(null);

  // Process vendor data
  const vendorData = invoices.reduce((acc, invoice) => {
    if (!acc[invoice.vendor]) {
      acc[invoice.vendor] = {
        name: invoice.vendor,
        totalValue: 0,
        invoiceCount: 0,
        approvedCount: 0,
        rejectedCount: 0,
        reviewCount: 0,
        totalIssues: 0,
        avgProcessingTime: 0,
        lastInvoiceDate: invoice.date,
        invoices: []
      };
    }
    
    acc[invoice.vendor].totalValue += invoice.amount;
    acc[invoice.vendor].invoiceCount += 1;
    acc[invoice.vendor].totalIssues += invoice.issues;
    acc[invoice.vendor].invoices.push(invoice);
    
    if (invoice.status === 'approved') acc[invoice.vendor].approvedCount += 1;
    if (invoice.status === 'rejected') acc[invoice.vendor].rejectedCount += 1;
    if (invoice.status === 'review_required') acc[invoice.vendor].reviewCount += 1;
    
    // Update last invoice date
    if (new Date(invoice.date) > new Date(acc[invoice.vendor].lastInvoiceDate)) {
      acc[invoice.vendor].lastInvoiceDate = invoice.date;
    }
    
    return acc;
  }, {});

  // Calculate additional metrics for each vendor
  Object.values(vendorData).forEach(vendor => {
    vendor.avgInvoiceValue = vendor.totalValue / vendor.invoiceCount;
    vendor.approvalRate = (vendor.approvedCount / vendor.invoiceCount) * 100;
    vendor.rejectionRate = (vendor.rejectedCount / vendor.invoiceCount) * 100;
    vendor.avgIssuesPerInvoice = vendor.totalIssues / vendor.invoiceCount;
    vendor.riskScore = (vendor.rejectionRate * 0.4) + (vendor.avgIssuesPerInvoice * 10 * 0.6);
    vendor.reliability = vendor.approvalRate > 80 ? 'High' : vendor.approvalRate > 60 ? 'Medium' : 'Low';
    vendor.relationship = (() => {
      const daysSinceFirst = Math.floor((new Date() - new Date(Math.min(...vendor.invoices.map(inv => new Date(inv.date))))) / (1000 * 60 * 60 * 24));
      if (daysSinceFirst > 365) return 'Long-term';
      if (daysSinceFirst > 90) return 'Established';
      return 'New';
    })();
  });

  // Convert to array and sort
  let vendorArray = Object.values(vendorData);
  
  // Apply risk filter
  if (showRiskOnly) {
    vendorArray = vendorArray.filter(vendor => vendor.riskScore > 20);
  }

  // Sort vendors
  vendorArray.sort((a, b) => {
    switch (sortBy) {
      case 'totalValue':
        return b.totalValue - a.totalValue;
      case 'invoiceCount':
        return b.invoiceCount - a.invoiceCount;
      case 'riskScore':
        return b.riskScore - a.riskScore;
      case 'approvalRate':
        return b.approvalRate - a.approvalRate;
      case 'avgInvoiceValue':
        return b.avgInvoiceValue - a.avgInvoiceValue;
      default:
        return 0;
    }
  });

  // Calculate summary statistics
  const totalVendors = vendorArray.length;
  const highRiskVendors = vendorArray.filter(v => v.riskScore > 50).length;
  const newVendors = vendorArray.filter(v => v.relationship === 'New').length;
  const avgVendorValue = vendorArray.reduce((sum, v) => sum + v.totalValue, 0) / totalVendors;

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const getRiskColor = (score) => {
    if (score > 50) return 'text-red-600 bg-red-100';
    if (score > 20) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  const getReliabilityColor = (reliability) => {
    switch (reliability) {
      case 'High': return 'text-green-600 bg-green-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'Low': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getRelationshipIcon = (relationship) => {
    switch (relationship) {
      case 'Long-term': return <Star className="w-4 h-4 text-yellow-500" />;
      case 'Established': return <Users className="w-4 h-4 text-blue-500" />;
      case 'New': return <Clock className="w-4 h-4 text-gray-500" />;
      default: return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Filter className="w-5 h-5 text-gray-400" />
            <div className="flex gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Sort By</label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="totalValue">Total Value</option>
                  <option value="invoiceCount">Invoice Count</option>
                  <option value="riskScore">Risk Score</option>
                  <option value="approvalRate">Approval Rate</option>
                  <option value="avgInvoiceValue">Avg Invoice Value</option>
                </select>
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={showRiskOnly}
                    onChange={(e) => setShowRiskOnly(e.target.checked)}
                    className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <span className="text-sm text-gray-700">Show High-Risk Only</span>
                </label>
              </div>
            </div>
          </div>
          <div className="text-sm text-gray-600">
            {vendorArray.length} vendor{vendorArray.length !== 1 ? 's' : ''} shown
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Users className="h-8 w-8 text-blue-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Vendors</p>
              <p className="text-2xl font-bold text-gray-900">{totalVendors}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <AlertTriangle className="h-8 w-8 text-red-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">High Risk</p>
              <p className="text-2xl font-bold text-gray-900">{highRiskVendors}</p>
              <p className="text-sm text-red-600">{((highRiskVendors / totalVendors) * 100).toFixed(1)}% of total</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Clock className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">New Vendors</p>
              <p className="text-2xl font-bold text-gray-900">{newVendors}</p>
              <p className="text-sm text-green-600">Requires monitoring</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <DollarSign className="h-8 w-8 text-purple-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg Vendor Value</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(avgVendorValue)}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Vendor List */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Vendor Performance Analysis</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Vendor
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Total Value
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Invoices
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Approval Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Risk Score
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Reliability
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Relationship
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {vendorArray.map((vendor, index) => (
                <tr key={vendor.name} className="hover:bg-gray-50 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {getRelationshipIcon(vendor.relationship)}
                      <div className="ml-2">
                        <div className="text-sm font-medium text-gray-900">{vendor.name}</div>
                        <div className="text-sm text-gray-500">
                          Avg: {formatCurrency(vendor.avgInvoiceValue)}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {formatCurrency(vendor.totalValue)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {vendor.invoiceCount}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                        <div
                          className="bg-green-600 h-2 rounded-full"
                          style={{ width: `${vendor.approvalRate}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-900">{vendor.approvalRate.toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getRiskColor(vendor.riskScore)}`}>
                      {vendor.riskScore.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getReliabilityColor(vendor.reliability)}`}>
                      {vendor.reliability}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {vendor.relationship}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button
                      onClick={() => setSelectedVendor(vendor)}
                      className="text-indigo-600 hover:text-indigo-900 flex items-center gap-1"
                    >
                      <Eye className="w-4 h-4" />
                      Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Vendor Detail Modal */}
      {selectedVendor && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-11/12 max-w-4xl shadow-lg rounded-md bg-white">
            <div className="flex justify-between items-center pb-4 border-b">
              <h3 className="text-lg font-medium text-gray-900">
                Vendor Analysis: {selectedVendor.name}
              </h3>
              <button
                onClick={() => setSelectedVendor(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>

            <div className="mt-6 space-y-6">
              {/* Vendor Overview */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Financial Summary</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Total Value:</span>
                      <span className="font-medium">{formatCurrency(selectedVendor.totalValue)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Avg Invoice:</span>
                      <span className="font-medium">{formatCurrency(selectedVendor.avgInvoiceValue)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Invoice Count:</span>
                      <span className="font-medium">{selectedVendor.invoiceCount}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Performance Metrics</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Approval Rate:</span>
                      <span className="font-medium">{selectedVendor.approvalRate.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Risk Score:</span>
                      <span className="font-medium">{selectedVendor.riskScore.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Avg Issues:</span>
                      <span className="font-medium">{selectedVendor.avgIssuesPerInvoice.toFixed(1)}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Relationship Info</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Type:</span>
                      <span className="font-medium">{selectedVendor.relationship}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Reliability:</span>
                      <span className="font-medium">{selectedVendor.reliability}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Last Invoice:</span>
                      <span className="font-medium">{selectedVendor.lastInvoiceDate}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recent Invoices */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Recent Invoices</h4>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {selectedVendor.invoices.slice(0, 10).map((invoice, index) => (
                    <div key={invoice.id} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                      <div>
                        <span className="font-medium">{invoice.id}</span>
                        <span className="text-gray-500 ml-2">{invoice.date}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{formatCurrency(invoice.amount)}</span>
                        <span className={`px-2 py-1 rounded text-xs ${invoice.status === 'approved' ? 'bg-green-100 text-green-800' : 
                          invoice.status === 'rejected' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'}`}>
                          {invoice.status.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-4 border-t mt-6">
              <button
                onClick={() => setSelectedVendor(null)}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VendorAnalytics;