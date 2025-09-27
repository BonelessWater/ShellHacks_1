import React from 'react';
import { CheckCircle, AlertTriangle, Activity, FileText } from 'lucide-react';

export const getStatusColor = (status) => {
  switch (status) {
    case 'approved': return 'text-green-600 bg-green-100';
    case 'rejected': return 'text-red-600 bg-red-100';
    case 'review_required': return 'text-yellow-600 bg-yellow-100';
    case 'processing': return 'text-blue-600 bg-blue-100';
    default: return 'text-gray-600 bg-gray-100';
  }
};

export const getStatusIcon = (status) => {
  switch (status) {
    case 'approved': return <CheckCircle className="h-5 w-5" />;
    case 'rejected': return <AlertTriangle className="h-5 w-5" />;
    case 'review_required': return <AlertTriangle className="h-5 w-5" />;
    case 'processing': return <Activity className="h-5 w-5 animate-spin" />;
    default: return <FileText className="h-5 w-5" />;
  }
};

export const formatCurrency = (amount) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(amount);
};