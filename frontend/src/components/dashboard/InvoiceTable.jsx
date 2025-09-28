import React, { useState } from 'react';
import { Eye, X, AlertTriangle, AlertCircle } from 'lucide-react';
import { getStatusColor, getStatusIcon } from '../../utils/formatters';

const InvoiceTable = ({ invoices, filteredInvoices }) => {
  const [selectedInvoice, setSelectedInvoice] = useState(null);

  const displayInvoices = filteredInvoices || invoices;

  const vendorName = (v) => {
    if (!v && v !== '') return 'Unknown Vendor';
    if (typeof v === 'string') return v;
    if (typeof v === 'object' && v !== null) return v.name || v.vendor || 'Unknown Vendor';
    return String(v);
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'high': return 'text-red-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'bg-red-500/20 text-red-400';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400';
      case 'low': return 'bg-blue-500/20 text-blue-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  // Enhanced mock data for demonstration
  const enhancedInvoices = displayInvoices.map(invoice => ({
    ...invoice,
    riskLevel: invoice.issues > 2 ? 'high' : invoice.issues > 0 ? 'medium' : 'low',
  description: vendorName(invoice.vendor).includes('ABC') ? 'Monthly office supplies' : 
        vendorName(invoice.vendor).includes('TechCorp') ? 'Software licensing fees' : 
                'Suspicious vendor transaction',
    detailedIssues: invoice.issues > 0 ? [
      { type: 'duplicate', severity: 'high', description: 'Potential duplicate payment detected' },
      { type: 'amount_anomaly', severity: 'medium', description: 'Amount exceeds typical range for vendor' },
      ...(invoice.issues > 2 ? [{ type: 'vendor_verification', severity: 'high', description: 'Vendor not in approved list' }] : [])
    ] : [],
    agentAnalysis: {
      fraudDetector: { score: invoice.issues * 25, reasoning: invoice.issues > 0 ? 'Multiple risk factors detected' : 'No fraud indicators found' },
      duplicateScanner: { score: invoice.issues > 1 ? 85 : 10, reasoning: invoice.issues > 1 ? 'Similar invoice found' : 'No duplicates detected' },
      vendorValidator: { score: invoice.issues > 2 ? 90 : 15, reasoning: invoice.issues > 2 ? 'Vendor verification failed' : 'Approved vendor' },
      amountValidator: { score: invoice.issues > 0 ? 60 : 20, reasoning: invoice.issues > 0 ? 'Amount deviation noted' : 'Amount within normal range' }
    },
    timeline: [
      { time: `${invoice.date} 09:23`, event: 'Invoice received', agent: 'System' },
      { time: `${invoice.date} 09:24`, event: 'Parallel verification initiated', agent: 'Orchestrator' },
      ...(invoice.issues > 0 ? [{ time: `${invoice.date} 09:25`, event: 'Issues detected', agent: 'Fraud Detector' }] : []),
      { time: `${invoice.date} 09:26`, event: `Status: ${invoice.status}`, agent: 'System' }
    ]
  }));

  return (
    <>
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Recent Invoices</h3>
          <p className="text-sm text-gray-500 mt-1">
            Showing {enhancedInvoices.length} invoice{enhancedInvoices.length !== 1 ? 's' : ''}
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Invoice ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Vendor
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Amount
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Risk Level
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Issues
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {enhancedInvoices.map((invoice) => (
                <tr key={invoice.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {invoice.id}
                  </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {vendorName(invoice.vendor)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${invoice.amount.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(invoice.status)}`}>
                      {getStatusIcon(invoice.status)}
                      <span className="ml-1 capitalize">{invoice.status.replace('_', ' ')}</span>
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <AlertTriangle className={`w-4 h-4 mr-1 ${getRiskColor(invoice.riskLevel)}`} />
                      <span className={`text-sm font-medium ${getRiskColor(invoice.riskLevel)}`}>
                        {invoice.riskLevel.charAt(0).toUpperCase() + invoice.riskLevel.slice(1)}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <div className="flex items-center">
                      <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                        <div
                          className="bg-green-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${invoice.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span>{Math.round(invoice.confidence * 100)}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {invoice.issues > 0 ? (
                      <div className="flex items-center">
                        <AlertCircle className="w-4 h-4 text-red-400 mr-1" />
                        <span className="text-red-600 font-medium">{invoice.issues}</span>
                      </div>
                    ) : (
                      <span className="text-green-600 font-medium">None</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button
                      onClick={() => setSelectedInvoice(invoice)}
                      className="inline-flex items-center px-3 py-1 border border-gray-300 rounded-md text-sm leading-4 font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors"
                    >
                      <Eye className="w-4 h-4 mr-1" />
                      View Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {enhancedInvoices.length === 0 && (
            <div className="text-center py-12">
              <AlertCircle className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No invoices found</h3>
              <p className="mt-1 text-sm text-gray-500">Try adjusting your filters or search terms.</p>
            </div>
          )}
        </div>
      </div>

      {/* Invoice Detail Modal */}
      {selectedInvoice && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-11/12 max-w-4xl shadow-lg rounded-md bg-white">
            {/* Modal Header */}
            <div className="flex justify-between items-center pb-4 border-b">
              <div>
                <h3 className="text-lg font-medium text-gray-900">
                  Invoice Details: {selectedInvoice.id}
                </h3>
                <p className="text-sm text-gray-500">Complete analysis and verification results</p>
              </div>
              <button
                onClick={() => setSelectedInvoice(null)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="mt-6 space-y-6 max-h-96 overflow-y-auto">
              {/* Invoice Summary */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="text-md font-medium text-gray-900 mb-3">Invoice Information</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Vendor:</span>
                        <span className="font-medium">{vendorName(selectedInvoice.vendor)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Amount:</span>
                      <span className="font-medium">${selectedInvoice.amount.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Date:</span>
                      <span className="font-medium">{selectedInvoice.date}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Status:</span>
                      <span className={`px-2 py-1 rounded text-sm ${getStatusColor(selectedInvoice.status)}`}>
                        {selectedInvoice.status.replace('_', ' ').toUpperCase()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Risk Level:</span>
                      <span className={`font-medium ${getRiskColor(selectedInvoice.riskLevel)}`}>
                        {selectedInvoice.riskLevel.toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="text-md font-medium text-gray-900 mb-3">Analysis Summary</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Confidence Level:</span>
                      <span className="font-medium">{Math.round(selectedInvoice.confidence * 100)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Issues Found:</span>
                      <span className="font-medium">{selectedInvoice.issues}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Processing Time:</span>
                      <span className="font-medium">1.2 seconds</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Verification Method:</span>
                      <span className="font-medium">AI Analysis</span>
                    </div>
                  </div>
                  
                  <div className="mt-3">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${selectedInvoice.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Issues Section */}
              {selectedInvoice.detailedIssues && selectedInvoice.detailedIssues.length > 0 && (
                <div>
                  <h4 className="text-md font-medium text-gray-900 mb-3">Issues Detected</h4>
                  <div className="space-y-3">
                    {selectedInvoice.detailedIssues.map((issue, index) => (
                      <div key={index} className="bg-gray-50 rounded-lg p-3">
                        <div className="flex justify-between items-start mb-2">
                          <span className="font-medium text-gray-900">
                            {issue.type.replace('_', ' ').toUpperCase()}
                          </span>
                          <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(issue.severity)}`}>
                            {issue.severity.toUpperCase()}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600">{issue.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Agent Analysis */}
              <div>
                <h4 className="text-md font-medium text-gray-900 mb-3">Agent Analysis</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {Object.entries(selectedInvoice.agentAnalysis).map(([agent, analysis]) => (
                    <div key={agent} className="bg-gray-50 rounded-lg p-3">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium text-gray-900">
                          {agent.replace(/([A-Z])/g, ' $1').trim()}
                        </span>
                        <span className={`font-bold text-sm ${
                          analysis.score > 70 ? 'text-red-600' : 
                          analysis.score > 30 ? 'text-yellow-600' : 'text-green-600'
                        }`}>
                          {analysis.score}%
                        </span>
                      </div>
                      <p className="text-xs text-gray-600 mb-2">{analysis.reasoning}</p>
                      <div className="w-full bg-gray-200 rounded-full h-1">
                        <div 
                          className={`h-1 rounded-full ${
                            analysis.score > 70 ? 'bg-red-400' : 
                            analysis.score > 30 ? 'bg-yellow-400' : 'bg-green-400'
                          }`}
                          style={{ width: `${analysis.score}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Processing Timeline */}
              <div>
                <h4 className="text-md font-medium text-gray-900 mb-3">Processing Timeline</h4>
                <div className="space-y-2">
                  {selectedInvoice.timeline.map((event, index) => (
                    <div key={index} className="flex items-center gap-3 p-2 bg-gray-50 rounded">
                      <div className="w-2 h-2 bg-blue-400 rounded-full flex-shrink-0"></div>
                      <div className="flex-1">
                        <span className="text-sm text-gray-900">{event.event}</span>
                        <div className="text-xs text-gray-500">by {event.agent}</div>
                      </div>
                      <span className="text-xs text-gray-500">{event.time}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="flex justify-end space-x-3 pt-4 border-t mt-6">
              <button
                onClick={() => setSelectedInvoice(null)}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400 transition-colors"
              >
                Close
              </button>
              <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors">
                Download Report
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default InvoiceTable;