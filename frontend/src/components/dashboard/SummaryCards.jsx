import React from 'react';
import { FileText, CheckCircle, AlertTriangle } from 'lucide-react';

const SummaryCards = ({ invoices }) => {
  const stats = {
    total: invoices.length,
    approved: invoices.filter(inv => inv.status === 'approved').length,
    needReview: invoices.filter(inv => inv.status === 'review_required').length,
    rejected: invoices.filter(inv => inv.status === 'rejected').length,
  };

  const cards = [
    {
      title: 'Total Invoices',
      value: stats.total,
      icon: FileText,
      color: 'text-blue-600'
    },
    {
      title: 'Approved',
      value: stats.approved,
      icon: CheckCircle,
      color: 'text-green-600'
    },
    {
      title: 'Need Review',
      value: stats.needReview,
      icon: AlertTriangle,
      color: 'text-yellow-600'
    },
    {
      title: 'Rejected',
      value: stats.rejected,
      icon: AlertTriangle,
      color: 'text-red-600'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
      {cards.map((card, index) => {
        const Icon = card.icon;
        return (
          <div key={index} className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <Icon className={`h-8 w-8 ${card.color}`} />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">{card.title}</p>
                <p className="text-2xl font-bold text-gray-900">{card.value}</p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default SummaryCards;