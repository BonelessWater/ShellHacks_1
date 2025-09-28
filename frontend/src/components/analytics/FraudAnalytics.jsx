import React, { useState } from 'react';
import { Shield, AlertTriangle, TrendingUp, DollarSign, Clock, Eye, Target, Activity } from 'lucide-react';
import { vendorName } from '../../utils/vendor';

const FraudAnalytics = ({ invoices }) => {
  const [timeFilter, setTimeFilter] = useState('all');
  const [selectedPattern, setSelectedPattern] = useState(null);

  // Filter invoices based on time
  const filteredInvoices = invoices.filter(invoice => {
    if (timeFilter === 'all') return true;
    const invoiceDate = new Date(invoice.date);
    const now = new Date();
    const daysDiff = (now - invoiceDate) / (1000 * 60 * 60 * 24);
    
    switch (timeFilter) {
      case 'week': return daysDiff <= 7;
      case 'month': return daysDiff <= 30;
      case 'quarter': return daysDiff <= 90;
      default: return true;
    }
  });

  // Fraud pattern detection
  const fraudPatterns = {
    roundAmounts: filteredInvoices.filter(inv => inv.amount % 100 === 0 || inv.amount % 1000 === 0),
    justUnderLimits: filteredInvoices.filter(inv => (inv.amount > 9000 && inv.amount < 10000) || (inv.amount > 4500 && inv.amount < 5000)),
    duplicateRisk: filteredInvoices.filter(inv => inv.issues > 2),
    suspiciousVendors: filteredInvoices.filter(inv => {
      const v = vendorName(inv.vendor).toLowerCase();
      return v.includes('suspicious') || v.includes('fraud');
    }),
    weekendSubmissions: filteredInvoices.filter(inv => {
      const day = new Date(inv.date).getDay();
      return day === 0 || day === 6; // Sunday = 0, Saturday = 6
    }),
    rapidSequence: (() => {
      // Group by vendor and check for rapid submissions
      const vendorGroups = filteredInvoices.reduce((acc, inv) => {
        const v = vendorName(inv.vendor);
        if (!acc[v]) acc[v] = [];
        acc[v].push(inv);
        return acc;
      }, {});
      
      return Object.values(vendorGroups).flatMap(group => {
        if (group.length < 2) return [];
        const sorted = group.sort((a, b) => new Date(a.date) - new Date(b.date));
        const rapid = [];
        for (let i = 1; i < sorted.length; i++) {
          const timeDiff = (new Date(sorted[i].date) - new Date(sorted[i-1].date)) / (1000 * 60 * 60 * 24);
          if (timeDiff <= 1) { // Same day or next day
            rapid.push(sorted[i], sorted[i-1]);
          }
        }
        return [...new Set(rapid)]; // Remove duplicates
      });
    })()
  };

  // Calculate fraud statistics
  const totalFraudulentInvoices = filteredInvoices.filter(inv => inv.status === 'rejected').length;
  const fraudRate = (totalFraudulentInvoices / filteredInvoices.length) * 100;
  const potentialSavings = filteredInvoices.filter(inv => inv.status === 'rejected').reduce((sum, inv) => sum + inv.amount, 0);
  const avgFraudAmount = potentialSavings / totalFraudulentInvoices || 0;

  // Risk score distribution
  const riskDistribution = {
    'Low Risk (0-1 issues)': filteredInvoices.filter(inv => inv.issues <= 1).length,
    'Medium Risk (2-3 issues)': filteredInvoices.filter(inv => inv.issues >= 2 && inv.issues <= 3).length,
    'High Risk (4+ issues)': filteredInvoices.filter(inv => inv.issues >= 4).length
  };

  // Detection accuracy metrics (simulated)
  const detectionMetrics = {
    truePositives: Math.floor(totalFraudulentInvoices * 0.95), // 95% accuracy
    falsePositives: Math.floor(filteredInvoices.filter(inv => inv.status === 'review_required').length * 0.1),
    falseNegatives: Math.floor(totalFraudulentInvoices * 0.05),
    trueNegatives: filteredInvoices.filter(inv => inv.status === 'approved').length
  };

  const precision = detectionMetrics.truePositives / (detectionMetrics.truePositives + detectionMetrics.falsePositives) * 100;
  const recall = detectionMetrics.truePositives / (detectionMetrics.truePositives + detectionMetrics.falseNegatives) * 100;
  const f1Score = 2 * (precision * recall) / (precision + recall);

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const getPatternRiskLevel = (patternSize, totalInvoices) => {
    const percentage = (patternSize / totalInvoices) * 100;
    if (percentage > 20) return { level: 'High', color: 'text-red-600 bg-red-100' };
    if (percentage > 10) return { level: 'Medium', color: 'text-yellow-600 bg-yellow-100' };
    return { level: 'Low', color: 'text-green-600 bg-green-100' };
  };

  return (
    <div className="space-y-6">
      {/* Time Filter */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center gap-4">
          <Shield className="w-5 h-5 text-blue-500" />
          <h3 className="text-lg font-medium text-gray-900">Fraud Detection Analysis</h3>
          <div className="ml-auto">
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
        </div>
      </div>

      {/* Key Fraud Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <AlertTriangle className="h-8 w-8 text-red-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Fraud Rate</p>
              <p className="text-2xl font-bold text-gray-900">{fraudRate.toFixed(1)}%</p>
              <p className="text-sm text-red-600">{totalFraudulentInvoices} fraudulent invoices</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <DollarSign className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Potential Savings</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(potentialSavings)}</p>
              <p className="text-sm text-green-600">Fraud prevented</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Target className="h-8 w-8 text-blue-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Detection Accuracy</p>
              <p className="text-2xl font-bold text-gray-900">{precision.toFixed(1)}%</p>
              <p className="text-sm text-blue-600">Precision rate</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Activity className="h-8 w-8 text-purple-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">F1 Score</p>
              <p className="text-2xl font-bold text-gray-900">{f1Score.toFixed(1)}%</p>
              <p className="text-sm text-purple-600">Overall performance</p>
            </div>
          </div>
        </div>
      </div>

      {/* Fraud Pattern Detection */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Fraud Pattern Detection</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(fraudPatterns).map(([patternName, invoices]) => {
            const risk = getPatternRiskLevel(invoices.length, filteredInvoices.length);
            const displayName = patternName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
            
            return (
              <div key={patternName} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-medium text-gray-900">{displayName}</h4>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${risk.color}`}>
                    {risk.level}
                  </span>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Detected:</span>
                    <span className="font-medium">{invoices.length} invoices</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Percentage:</span>
                    <span className="font-medium">{((invoices.length / filteredInvoices.length) * 100).toFixed(1)}%</span>
                  </div>
                  {invoices.length > 0 && (
                    <button
                      onClick={() => setSelectedPattern({ name: displayName, invoices })}
                      className="w-full mt-2 px-3 py-1 bg-indigo-50 text-indigo-700 rounded text-sm hover:bg-indigo-100 transition-colors"
                    >
                      View Details
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Risk Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Risk Score Distribution</h3>
          <div className="space-y-4">
            {Object.entries(riskDistribution).map(([riskLevel, count]) => {
              const percentage = (count / filteredInvoices.length) * 100;
              return (
                <div key={riskLevel} className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">{riskLevel}</span>
                  <div className="flex items-center gap-3">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-green-500 to-red-500 h-2 rounded-full"
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900 w-16">
                      {count} ({percentage.toFixed(1)}%)
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Detection Performance Metrics</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Precision (True Positive Rate)</span>
              <span className="text-sm font-medium text-gray-900">{precision.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Recall (Sensitivity)</span>
              <span className="text-sm font-medium text-gray-900">{recall.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">F1 Score</span>
              <span className="text-sm font-medium text-gray-900">{f1Score.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">False Positive Rate</span>
              <span className="text-sm font-medium text-gray-900">
                {((detectionMetrics.falsePositives / (detectionMetrics.falsePositives + detectionMetrics.trueNegatives)) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Fraud Trends */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Fraud Prevention Impact</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600">{formatCurrency(potentialSavings)}</div>
            <div className="text-sm text-gray-600 mt-1">Total Fraud Prevented</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">{formatCurrency(avgFraudAmount)}</div>
            <div className="text-sm text-gray-600 mt-1">Average Fraud Amount</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600">
              {Math.round((detectionMetrics.truePositives / (detectionMetrics.truePositives + detectionMetrics.falseNegatives)) * 100)}%
            </div>
            <div className="text-sm text-gray-600 mt-1">Fraud Detection Rate</div>
          </div>
        </div>
      </div>

      {/* Pattern Detail Modal */}
      {selectedPattern && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-11/12 max-w-4xl shadow-lg rounded-md bg-white">
            <div className="flex justify-between items-center pb-4 border-b">
              <h3 className="text-lg font-medium text-gray-900">
                Fraud Pattern: {selectedPattern.name}
              </h3>
              <button
                onClick={() => setSelectedPattern(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>

            <div className="mt-6">
              <p className="text-sm text-gray-600 mb-4">
                Found {selectedPattern.invoices.length} invoices matching this fraud pattern
              </p>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {selectedPattern.invoices.map((invoice, index) => (
                  <div key={invoice.id} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <div>
                      <span className="font-medium">{invoice.id}</span>
                      <span className="text-gray-500 ml-2">{vendorName(invoice.vendor)}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="font-medium">{formatCurrency(invoice.amount)}</span>
                      <span className="text-sm text-gray-500">{invoice.date}</span>
                      <span className={`px-2 py-1 rounded text-xs ${
                        invoice.status === 'rejected' ? 'bg-red-100 text-red-800' :
                        invoice.status === 'review_required' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {invoice.status.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-end pt-4 border-t mt-6">
              <button
                onClick={() => setSelectedPattern(null)}
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

export default FraudAnalytics;