// frontend/src/utils/mapAnalysisResult.js
export function mapBackendToUI(payload) {
  if (!payload) return null;

  // Accept either /api/upload shape (top-level fraud_analysis)
  // or /api/invoices/analyze shape (data contains the analysis)
  const fa = payload.fraud_analysis || payload.data || payload; // try them in order
  const meta = fa.metadata || payload.metadata || {};
  const data = payload.data || {};

  // Risk: backend may be 0–10 (our /api/upload) OR already 0–10 from /api/invoices/analyze
  let rawRisk = Number(fa.overall_risk_score);
  if (!Number.isFinite(rawRisk)) rawRisk = 0;
  const riskPct = Math.max(0, Math.min(100, rawRisk * 10)); // UI expects percentage

  // Recommendation -> UI status
  const rec = String(fa.recommendation || '').toUpperCase();
  let status = 'review_required';
  if (rec.includes('APPROV')) status = 'approved';
  else if (rec.includes('REJECT')) status = 'rejected';
  else if (rec.includes('REVIEW')) status = 'review_required';

  // Confidence may be 'N/A', 0–1 or 0–100
  let conf = fa.confidence;
  if (conf === 'N/A' || conf == null || conf === '') conf = 0.7;
  conf = Number(conf);
  if (!Number.isFinite(conf)) conf = 0.7;
  if (conf > 1) conf = conf / 100;

  // Invoice id
  const invoice_id =
    data.saved_as ||
    fa.invoice_id ||
    data.filename ||
    `invoice_${Date.now()}`;

  // Agent results: prefer fa.agent_results if present
  const agent_results = Array.isArray(fa.agent_results) ? fa.agent_results : [];

  // processing time
  const processing_time =
    fa.processing_time ??
    data.analysis_duration_seconds ??
    payload.performance?.actual_time_seconds ??
    null;

  const analysis_summary =
    fa.analysis_summary || 'Comprehensive fraud analysis completed.';

  const red_flags = Array.isArray(fa.red_flags) ? fa.red_flags : [];

  const metadata = {
    total_agents_executed:
      meta.total_agents_executed ??
      agent_results.length ??
      0,
    successful_agents:
      meta.successful_agents ??
      agent_results.filter(a => a.status === 'completed').length ??
      0,
    failed_agents:
      meta.failed_agents ??
      agent_results.filter(a => a.status && a.status !== 'completed').length ??
      0,
    analysis_timestamp:
      meta.analysis_timestamp ||
      data.upload_time ||
      new Date().toISOString(),
  };

  return {
    status,
    invoice_id,
    overall_risk_score: riskPct,
    confidence: conf,
    red_flags,
    processing_time,
    recommendation: fa.recommendation || 'Review required',
    analysis_summary,
    agent_results,
    metadata,
  };
}
