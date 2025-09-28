import React, { useEffect, useState } from 'react';
import { apiService } from '../../services/api';

const OrchestratorStats = ({ invoices = [] }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const computeStats = async () => {
      if (!invoices || invoices.length === 0) return;
      try {
        setLoading(true);
        // Ask backend to score the invoices using agents (or fallback)
        const resp = await apiService.scoreInvoices(invoices.slice(0, 50));
        const results = resp.results || [];

        // Aggregate
        const confidences = results.map(r => (r.score && r.score.confidence) || 0);
        const avgConfidence = confidences.reduce((s, v) => s + v, 0) / (confidences.length || 1);

        const riskCounts = results.reduce((acc, r) => {
          const risk = (r.score && r.score.risk) || 'unknown';
          acc[risk] = (acc[risk] || 0) + 1;
          return acc;
        }, {});

        setStats({ avgConfidence: Number(avgConfidence.toFixed(3)), riskCounts, total: results.length });
      } catch (err) {
        console.error('OrchestratorStats error', err);
        setError(err.message || String(err));
      } finally {
        setLoading(false);
      }
    };

    computeStats();
  }, [invoices]);

  if (!invoices || invoices.length === 0) return null;

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Orchestrator Statistics</h3>
      {loading && <div className="text-sm text-gray-500">Computing agent scores...</div>}
      {error && <div className="text-sm text-red-600">Error: {error}</div>}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <div className="text-sm text-gray-600">Average Confidence</div>
            <div className="text-2xl font-bold">{(stats.avgConfidence * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Total Scored</div>
            <div className="text-2xl font-bold">{stats.total}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Risk Distribution</div>
            <div className="text-base">
              {Object.entries(stats.riskCounts).map(([k,v]) => (
                <div key={k} className="flex justify-between">
                  <span className="capitalize">{k}</span>
                  <span className="font-medium">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OrchestratorStats;
