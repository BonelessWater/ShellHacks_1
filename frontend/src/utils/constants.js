export const INVOICE_STATUS = {
  PROCESSING: 'processing',
  APPROVED: 'approved',
  REJECTED: 'rejected',
  REVIEW_REQUIRED: 'review_required'
};

export const RISK_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical'
};

export const AGENT_TYPES = {
  ORCHESTRATOR: 'orchestrator',
  VALIDATION: 'validation',
  FRAUD_DETECTION: 'fraud_detection',
  ANALYTICS: 'analytics',
  COMPLIANCE: 'compliance'
};

export const FILE_TYPES = {
  PDF: '.pdf',
  PNG: '.png',
  JPG: '.jpg',
  JPEG: '.jpeg',
  XLSX: '.xlsx',
  XLS: '.xls'
};

export const API_ENDPOINTS = {
  INVOICES: '/api/invoices',
  UPLOAD: '/api/invoices/upload',
  SYSTEM_STATUS: '/api/system/status',
  ANALYTICS: '/api/analytics',
  AGENT_CONFIG: '/api/agents/config'
};