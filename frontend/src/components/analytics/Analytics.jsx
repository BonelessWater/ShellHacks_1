import React, { useState } from 'react';
import { TrendingUp, Users, FileText, Shield, Activity, DollarSign, Clock, AlertTriangle, CheckCircle } from 'lucide-react';
import OrchestratorStats from './OrchestratorStats';
import { vendorName } from '../../utils/vendor';

const Analytics = ({ systemStatus, invoices = [] }) => {
  const [activeTab, setActiveTab] = useState('business');

  const tabs = [
    { id: 'business', label: 'Business Metrics', icon: TrendingUp },
    { id: 'invoices', label: 'Invoice Analytics', icon: FileText },
    { id: 'vendors', label: 'Vendor Analytics', icon: Users },
    { id: 'fraud', label: 'Fraud Analytics', icon: Shield }
  ];

  // Calculate data using the same structure as Dashboard
  const totalInvoices = invoices.length;
  const totalAmount = invoices.reduce((sum, invoice) => sum + (invoice.amount || 0), 0);
  const averageAmount = totalAmount / totalInvoices || 0;
  const rejectedInvoices = invoices.filter(invoice => invoice.status === 'rejected').length;
  const reviewRequiredInvoices = invoices.filter(invoice => invoice.status === 'review_required').length;
  const fraudulentInvoices = rejectedInvoices + reviewRequiredInvoices; // Combine problematic invoices
  const fraudRate = (fraudulentInvoices / totalInvoices) * 100 || 0;

  const BusinessMetrics = () => (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <FileText className="h-8 w-8 text-blue-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">Total Invoices</dt>
                <dd className="text-lg font-medium text-gray-900">{totalInvoices}</dd>
              </dl>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <DollarSign className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">Total Value</dt>
                <dd className="text-lg font-medium text-gray-900">${totalAmount.toLocaleString()}</dd>
              </dl>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <TrendingUp className="h-8 w-8 text-purple-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">Average Amount</dt>
                <dd className="text-lg font-medium text-gray-900">${averageAmount.toFixed(2)}</dd>
              </dl>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Shield className="h-8 w-8 text-red-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">Fraud Rate</dt>
                <dd className="text-lg font-medium text-gray-900">{fraudRate.toFixed(1)}%</dd>
              </dl>
            </div>
          </div>
        </div>
      </div>

      {/* Monthly Trends Chart Placeholder */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Monthly Processing Trends</h3>
        <div className="h-64 bg-gray-50 rounded flex items-center justify-center">
          <p className="text-gray-500">Chart visualization would go here</p>
        </div>
      </div>
    </div>
  );

  const InvoiceAnalytics = () => (
    <div className="space-y-6">
      {/* Status Distribution */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Invoice Status Distribution</h3>
        <div className="space-y-4">
          {['approved', 'processing', 'review_required', 'rejected'].map(status => {
            const count = invoices.filter(invoice => invoice.status === status).length;
            const percentage = ((count / totalInvoices) * 100) || 0;
            const displayLabel = status === 'review_required' ? 'Review Required' : 
                               status.charAt(0).toUpperCase() + status.slice(1);
            return (
              <div key={status} className="flex items-center">
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-900">{displayLabel}</span>
                    <span className="text-sm text-gray-500">{count} ({percentage.toFixed(1)}%)</span>
                  </div>
                  <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        status === 'approved' ? 'bg-green-600' :
                        status === 'processing' ? 'bg-blue-600' :
                        status === 'review_required' ? 'bg-yellow-600' : 'bg-red-600'
                      }`}
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Processing Time Analysis */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
          <Clock className="w-5 h-5 mr-2" />
          Processing Time Analysis
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">2.3s</div>
            <div className="text-sm text-gray-500">Average Processing</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">1.1s</div>
            <div className="text-sm text-gray-500">Fastest</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">5.7s</div>
            <div className="text-sm text-gray-500">Slowest</div>
          </div>
        </div>
      </div>
    </div>
  );

  const VendorAnalytics = () => {
    const vendorStats = invoices.reduce((acc, invoice) => {
      const vendor = vendorName(invoice.vendor);
      if (!acc[vendor]) {
        acc[vendor] = { count: 0, totalAmount: 0, problemCount: 0 };
      }
      acc[vendor].count++;
      acc[vendor].totalAmount += invoice.amount || 0;
      if (invoice.status === 'rejected' || invoice.status === 'review_required') {
        acc[vendor].problemCount++;
      }
      return acc;
    }, {});

    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Top Vendors by Volume</h3>
          <div className="space-y-4">
            {Object.entries(vendorStats)
              .sort(([,a], [,b]) => b.count - a.count)
              .slice(0, 10)
              .map(([vendor, stats]) => (
                <div key={vendor} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div>
                    <div className="font-medium text-gray-900">{vendor}</div>
                    <div className="text-sm text-gray-500">
                      {stats.count} invoices • ${stats.totalAmount.toLocaleString()}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-medium ${
                      stats.problemCount > 0 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {stats.problemCount} issues
                    </div>
                    <div className="text-xs text-gray-500">
                      {((stats.problemCount / stats.count) * 100).toFixed(1)}% problem rate
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>
    );
  };

  const FraudAnalytics = () => (
    <div className="space-y-6">
      {/* Orchestrator stats: agent-based confidence & distributions */}
      <OrchestratorStats invoices={invoices} />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-red-600" />
            Fraud Detection Summary
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Flagged</span>
              <span className="font-medium">{fraudulentInvoices}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Fraud Rate</span>
              <span className="font-medium text-red-600">{fraudRate.toFixed(2)}%</span>
            </div>
                <div className="flex justify-between">
              <span className="text-gray-600">Amount at Risk</span>
              <span className="font-medium">
                ${invoices.filter(invoice => invoice.status === 'rejected' || invoice.status === 'review_required')
                  .reduce((sum, invoice) => sum + (invoice.amount || 0), 0).toLocaleString()}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Common Fraud Patterns</h3>
          <div className="space-y-2">
            <div className="text-sm text-gray-600">• Duplicate invoice numbers</div>
            <div className="text-sm text-gray-600">• Unusual vendor activity</div>
            <div className="text-sm text-gray-600">• Amount discrepancies</div>
            <div className="text-sm text-gray-600">• Low confidence scores</div>
          </div>
        </div>
      </div>
    </div>
  );

  const PerformanceMonitoring = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">AI Agent Performance</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">OCR Accuracy</span>
              <span className="font-medium text-green-600">97.8%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Fraud Detection</span>
              <span className="font-medium text-green-600">94.2%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Processing Speed</span>
              <span className="font-medium text-blue-600">2.3s avg</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">System Health</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">API Status</span>
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Database</span>
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Queue</span>
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Resource Usage</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">CPU Usage</span>
              <span className="font-medium">23%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Memory</span>
              <span className="font-medium">67%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Storage</span>
              <span className="font-medium">45%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'business':
        return <BusinessMetrics />;
      case 'invoices':
        return <InvoiceAnalytics />;
      case 'vendors':
        return <VendorAnalytics />;
      case 'fraud':
        return <FraudAnalytics />;
      default:
        return <BusinessMetrics />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="bg-white shadow rounded-lg">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8 px-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-5 h-5 mr-2" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Content Area */}
      <div>
        {renderContent()}
      </div>
    </div>
  );
};

export default Analytics;