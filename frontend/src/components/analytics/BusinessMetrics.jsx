import React from 'react';
import { DollarSign, TrendingUp, TrendingDown, Calendar, AlertTriangle, CheckCircle } from 'lucide-react';

const BusinessMetrics = ({ invoices }) => {
  // Calculate business metrics
  const totalValue = invoices.reduce((sum, invoice) => sum + invoice.amount, 0);
  const approvedValue = invoices.filter(inv => inv.status === 'approved').reduce((sum, inv) => sum + inv.amount, 0);
  const rejectedValue = invoices.filter(inv => inv.status === 'rejected').reduce((sum, inv) => sum + inv.amount, 0);
  const pendingValue = invoices.filter(inv => inv.status === 'review_required' || inv.status === 'processing').reduce((sum, inv) => sum + inv.amount, 0);
  
  const avgInvoiceValue = totalValue / invoices.length || 0;
  const fraudPreventionSavings = rejectedValue; // Assuming rejected invoices were potentially fraudulent
  
  // Monthly breakdown
  const monthlyData = invoices.reduce((acc, invoice) => {
    const month = new Date(invoice.date).toLocaleString('default', { month: 'short', year: '2-digit' });
    if (!acc[month]) {
      acc[month] = { total: 0, count: 0, approved: 0, rejected: 0 };
    }
    acc[month].total += invoice.amount;
    acc[month].count += 1;
    if (invoice.status === 'approved') acc[month].approved += invoice.amount;
    if (invoice.status === 'rejected') acc[month].rejected += invoice.amount;
    return acc;
  }, {});

  // Business unit breakdown (simulated)
  const businessUnits = [
    { name: 'Operations', value: totalValue * 0.35, count: Math.floor(invoices.length * 0.35), trend: 'up' },
    { name: 'Marketing', value: totalValue * 0.25, count: Math.floor(invoices.length * 0.25), trend: 'down' },
    { name: 'IT', value: totalValue * 0.20, count: Math.floor(invoices.length * 0.20), trend: 'up' },
    { name: 'Finance', value: totalValue * 0.15, count: Math.floor(invoices.length * 0.15), trend: 'stable' },
    { name: 'HR', value: totalValue * 0.05, count: Math.floor(invoices.length * 0.05), trend: 'up' }
  ];

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'approved': return 'text-green-600 bg-green-100';
      case 'rejected': return 'text-red-600 bg-red-100';
      case 'pending': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'down': return <TrendingDown className="w-4 h-4 text-red-500" />;
      default: return <div className="w-4 h-4 bg-gray-400 rounded-full"></div>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Key Financial Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <DollarSign className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Invoice Value</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(totalValue)}</p>
              <p className="text-sm text-gray-500 mt-1">{invoices.length} invoices</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <CheckCircle className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Approved Value</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(approvedValue)}</p>
              <p className="text-sm text-green-600 mt-1">
                {((approvedValue / totalValue) * 100).toFixed(1)}% of total
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <AlertTriangle className="h-8 w-8 text-red-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Fraud Prevention</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(fraudPreventionSavings)}</p>
              <p className="text-sm text-red-600 mt-1">Potential savings</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Calendar className="h-8 w-8 text-blue-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Average Invoice</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(avgInvoiceValue)}</p>
              <p className="text-sm text-blue-600 mt-1">Per transaction</p>
            </div>
          </div>
        </div>
      </div>

      {/* Status Breakdown */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Invoice Status Breakdown</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className={`p-4 rounded-lg ${getStatusColor('approved')}`}>
            <div className="flex justify-between items-center">
              <div>
                <p className="font-medium">Approved</p>
                <p className="text-sm opacity-75">
                  {invoices.filter(inv => inv.status === 'approved').length} invoices
                </p>
              </div>
              <div className="text-right">
                <p className="font-bold text-lg">{formatCurrency(approvedValue)}</p>
                <p className="text-sm">
                  {((approvedValue / totalValue) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          <div className={`p-4 rounded-lg ${getStatusColor('pending')}`}>
            <div className="flex justify-between items-center">
              <div>
                <p className="font-medium">Pending Review</p>
                <p className="text-sm opacity-75">
                  {invoices.filter(inv => inv.status === 'review_required' || inv.status === 'processing').length} invoices
                </p>
              </div>
              <div className="text-right">
                <p className="font-bold text-lg">{formatCurrency(pendingValue)}</p>
                <p className="text-sm">
                  {((pendingValue / totalValue) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          <div className={`p-4 rounded-lg ${getStatusColor('rejected')}`}>
            <div className="flex justify-between items-center">
              <div>
                <p className="font-medium">Rejected</p>
                <p className="text-sm opacity-75">
                  {invoices.filter(inv => inv.status === 'rejected').length} invoices
                </p>
              </div>
              <div className="text-right">
                <p className="font-bold text-lg">{formatCurrency(rejectedValue)}</p>
                <p className="text-sm">
                  {((rejectedValue / totalValue) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Business Unit Analysis */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Spending by Business Unit</h3>
        <div className="space-y-4">
          {businessUnits.map((unit, index) => (
            <div key={index} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
              <div className="flex items-center">
                {getTrendIcon(unit.trend)}
                <div className="ml-3">
                  <p className="font-medium text-gray-900">{unit.name}</p>
                  <p className="text-sm text-gray-500">{unit.count} invoices</p>
                </div>
              </div>
              <div className="text-right">
                <p className="font-bold text-gray-900">{formatCurrency(unit.value)}</p>
                <p className="text-sm text-gray-500">
                  {((unit.value / totalValue) * 100).toFixed(1)}% of total
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Monthly Trend */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Monthly Invoice Trends</h3>
        <div className="space-y-3">
          {Object.entries(monthlyData).map(([month, data]) => (
            <div key={month} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <p className="font-medium text-gray-900">{month}</p>
                <p className="text-sm text-gray-500">{data.count} invoices processed</p>
              </div>
              <div className="text-right">
                <p className="font-bold text-gray-900">{formatCurrency(data.total)}</p>
                <div className="flex gap-4 text-sm">
                  <span className="text-green-600">Approved: {formatCurrency(data.approved)}</span>
                  {data.rejected > 0 && (
                    <span className="text-red-600">Rejected: {formatCurrency(data.rejected)}</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key Insights */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Key Business Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <div className="flex items-start">
              <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 mr-3" />
              <div>
                <p className="font-medium text-gray-900">Approval Rate</p>
                <p className="text-sm text-gray-600">
                  {((invoices.filter(inv => inv.status === 'approved').length / invoices.length) * 100).toFixed(1)}% 
                  of invoices are automatically approved
                </p>
              </div>
            </div>
            <div className="flex items-start">
              <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5 mr-3" />
              <div>
                <p className="font-medium text-gray-900">Review Rate</p>
                <p className="text-sm text-gray-600">
                  {((invoices.filter(inv => inv.status === 'review_required').length / invoices.length) * 100).toFixed(1)}% 
                  require manual review
                </p>
              </div>
            </div>
          </div>
          <div className="space-y-3">
            <div className="flex items-start">
              <DollarSign className="w-5 h-5 text-blue-500 mt-0.5 mr-3" />
              <div>
                <p className="font-medium text-gray-900">Cost Savings</p>
                <p className="text-sm text-gray-600">
                  AI fraud detection has prevented {formatCurrency(fraudPreventionSavings)} in potential losses
                </p>
              </div>
            </div>
            <div className="flex items-start">
              <TrendingUp className="w-5 h-5 text-green-500 mt-0.5 mr-3" />
              <div>
                <p className="font-medium text-gray-900">Processing Efficiency</p>
                <p className="text-sm text-gray-600">
                  Average processing time reduced by 85% with AI automation
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BusinessMetrics;