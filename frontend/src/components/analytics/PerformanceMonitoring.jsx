import React, { useState, useEffect } from 'react';
import { Activity, Cpu, Clock, TrendingUp, AlertCircle, CheckCircle, Zap, BarChart3 } from 'lucide-react';

const PerformanceMonitoring = ({ systemStatus }) => {
  const [timeRange, setTimeRange] = useState('24h');
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [performanceHistory, setPerformanceHistory] = useState([]);

  // Generate mock performance history
  useEffect(() => {
    const generateHistory = () => {
      const hours = timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 720;
      const interval = timeRange === '24h' ? 1 : timeRange === '7d' ? 6 : 24;
      
      return Array.from({ length: Math.floor(hours / interval) }, (_, i) => ({
        timestamp: new Date(Date.now() - (hours - i * interval) * 60 * 60 * 1000),
        fraudDetector: {
          load: Math.random() * 30 + 60,
          accuracy: Math.random() * 5 + 95,
          responseTime: Math.random() * 0.5 + 0.3,
          errors: Math.floor(Math.random() * 3)
        },
        duplicateScanner: {
          load: Math.random() * 40 + 40,
          accuracy: Math.random() * 3 + 97,
          responseTime: Math.random() * 0.3 + 0.2,
          errors: Math.floor(Math.random() * 2)
        },
        vendorValidator: {
          load: Math.random() * 35 + 45,
          accuracy: Math.random() * 4 + 94,
          responseTime: Math.random() * 0.4 + 0.4,
          errors: Math.floor(Math.random() * 2)
        },
        amountValidator: {
          load: Math.random() * 25 + 50,
          accuracy: Math.random() * 2 + 98,
          responseTime: Math.random() * 0.2 + 0.1,
          errors: Math.floor(Math.random() * 1)
        }
      }));
    };

    setPerformanceHistory(generateHistory());
  }, [timeRange]);

  // Current agent performance (enhanced from systemStatus)
  const agentPerformance = {
    fraudDetector: {
      name: 'Fraud Detection Agent',
      status: 'active',
      load: systemStatus?.agents?.fraud_detection?.load || 85,
      accuracy: 97.3,
      responseTime: 0.8,
      uptime: 99.9,
      tasksCompleted: 1247,
      errorsToday: 2,
      modelVersion: 'v2.1.3',
      lastUpdate: '2024-01-15',
      specialty: 'Pattern recognition and anomaly detection'
    },
    duplicateScanner: {
      name: 'Duplicate Detection Agent',
      status: 'active', 
      load: systemStatus?.agents?.validation?.load || 72,
      accuracy: 99.1,
      responseTime: 0.4,
      uptime: 99.8,
      tasksCompleted: 892,
      errorsToday: 1,
      modelVersion: 'v1.8.2',
      lastUpdate: '2024-01-12',
      specialty: 'Invoice similarity and duplicate identification'
    },
    vendorValidator: {
      name: 'Vendor Verification Agent',
      status: 'active',
      load: systemStatus?.agents?.orchestrator?.load || 68,
      accuracy: 96.7,
      responseTime: 1.2,
      uptime: 99.7,
      tasksCompleted: 654,
      errorsToday: 3,
      modelVersion: 'v1.5.1',
      lastUpdate: '2024-01-10',
      specialty: 'Vendor database verification and risk assessment'
    },
    amountValidator: {
      name: 'Amount Validation Agent',
      status: 'active',
      load: systemStatus?.agents?.analytics?.load || 45,
      accuracy: 98.9,
      responseTime: 0.3,
      uptime: 99.9,
      tasksCompleted: 1156,
      errorsToday: 0,
      modelVersion: 'v2.0.1',
      lastUpdate: '2024-01-14',
      specialty: 'Mathematical validation and pricing verification'
    }
  };

  // System-wide performance metrics
  const systemMetrics = {
    totalRequests: Object.values(agentPerformance).reduce((sum, agent) => sum + agent.tasksCompleted, 0),
    avgResponseTime: Object.values(agentPerformance).reduce((sum, agent) => sum + agent.responseTime, 0) / Object.keys(agentPerformance).length,
    systemUptime: Math.min(...Object.values(agentPerformance).map(agent => agent.uptime)),
    totalErrors: Object.values(agentPerformance).reduce((sum, agent) => sum + agent.errorsToday, 0),
    avgAccuracy: Object.values(agentPerformance).reduce((sum, agent) => sum + agent.accuracy, 0) / Object.keys(agentPerformance).length
  };

  const getStatusColor = (status, load) => {
    if (status !== 'active') return 'text-red-600 bg-red-100';
    if (load > 90) return 'text-red-600 bg-red-100';
    if (load > 75) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  const getPerformanceGrade = (accuracy) => {
    if (accuracy >= 98) return { grade: 'A+', color: 'text-green-600' };
    if (accuracy >= 95) return { grade: 'A', color: 'text-green-600' };
    if (accuracy >= 90) return { grade: 'B', color: 'text-yellow-600' };
    if (accuracy >= 85) return { grade: 'C', color: 'text-orange-600' };
    return { grade: 'D', color: 'text-red-600' };
  };

  const formatUptime = (uptime) => {
    return `${uptime.toFixed(2)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Activity className="w-5 h-5 text-blue-500" />
            <h3 className="text-lg font-medium text-gray-900">AI Agent Performance Monitoring</h3>
          </div>
          <div>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:ring-indigo-500 focus:border-indigo-500"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
        </div>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <BarChart3 className="h-8 w-8 text-blue-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Requests</p>
              <p className="text-2xl font-bold text-gray-900">{systemMetrics.totalRequests.toLocaleString()}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Clock className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg Response Time</p>
              <p className="text-2xl font-bold text-gray-900">{systemMetrics.avgResponseTime.toFixed(2)}s</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <CheckCircle className="h-8 w-8 text-purple-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">System Uptime</p>
              <p className="text-2xl font-bold text-gray-900">{formatUptime(systemMetrics.systemUptime)}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <TrendingUp className="h-8 w-8 text-indigo-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg Accuracy</p>
              <p className="text-2xl font-bold text-gray-900">{systemMetrics.avgAccuracy.toFixed(1)}%</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <AlertCircle className="h-8 w-8 text-red-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Errors Today</p>
              <p className="text-2xl font-bold text-gray-900">{systemMetrics.totalErrors}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Agent Performance Table */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Individual Agent Performance</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Load</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Response Time</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Uptime</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Tasks Today</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {Object.entries(agentPerformance).map(([agentId, agent]) => {
                const grade = getPerformanceGrade(agent.accuracy);
                return (
                  <tr key={agentId} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Cpu className="w-5 h-5 text-gray-400 mr-3" />
                        <div>
                          <div className="text-sm font-medium text-gray-900">{agent.name}</div>
                          <div className="text-sm text-gray-500">{agent.modelVersion}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(agent.status, agent.load)}`}>
                        {agent.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className={`h-2 rounded-full ${agent.load > 90 ? 'bg-red-500' : agent.load > 75 ? 'bg-yellow-500' : 'bg-green-500'}`}
                            style={{ width: `${agent.load}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-900">{agent.load}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <span className="text-sm text-gray-900">{agent.accuracy}%</span>
                        <span className={`ml-2 text-xs font-bold ${grade.color}`}>({grade.grade})</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {agent.responseTime}s
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatUptime(agent.uptime)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {agent.tasksCompleted.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button
                        onClick={() => setSelectedAgent(agent)}
                        className="text-indigo-600 hover:text-indigo-900"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Load Distribution Over Time</h3>
          <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-500 text-sm">Load trend visualization would be displayed here</p>
              <p className="text-xs text-gray-400 mt-1">Real-time performance monitoring active</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Response Time Trends</h3>
          <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <Clock className="w-12 h-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-500 text-sm">Response time analytics would be displayed here</p>
              <p className="text-xs text-gray-400 mt-1">Average response time: {systemMetrics.avgResponseTime.toFixed(2)}s</p>
            </div>
          </div>
        </div>
      </div>

      {/* Model Information */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">AI Model Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Object.entries(agentPerformance).map(([agentId, agent]) => (
            <div key={agentId} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium text-gray-900">{agent.name}</h4>
                <Zap className="w-4 h-4 text-yellow-500" />
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Version:</span>
                  <span className="font-medium">{agent.modelVersion}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Updated:</span>
                  <span className="font-medium">{agent.lastUpdate}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Errors Today:</span>
                  <span className={`font-medium ${agent.errorsToday === 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {agent.errorsToday}
                  </span>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-3">{agent.specialty}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Agent Detail Modal */}
      {selectedAgent && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-11/12 max-w-4xl shadow-lg rounded-md bg-white">
            <div className="flex justify-between items-center pb-4 border-b">
              <h3 className="text-lg font-medium text-gray-900">
                {selectedAgent.name} - Detailed Performance
              </h3>
              <button
                onClick={() => setSelectedAgent(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>

            <div className="mt-6 space-y-6">
              {/* Agent Overview */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Performance Metrics</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Current Load:</span>
                      <span className="font-medium">{selectedAgent.load}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Accuracy:</span>
                      <span className="font-medium">{selectedAgent.accuracy}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Response Time:</span>
                      <span className="font-medium">{selectedAgent.responseTime}s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Uptime:</span>
                      <span className="font-medium">{formatUptime(selectedAgent.uptime)}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Model Information</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Version:</span>
                      <span className="font-medium">{selectedAgent.modelVersion}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Last Update:</span>
                      <span className="font-medium">{selectedAgent.lastUpdate}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Status:</span>
                      <span className="font-medium capitalize">{selectedAgent.status}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Activity Summary</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Tasks Today:</span>
                      <span className="font-medium">{selectedAgent.tasksCompleted}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Errors Today:</span>
                      <span className={`font-medium ${selectedAgent.errorsToday === 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {selectedAgent.errorsToday}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Success Rate:</span>
                      <span className="font-medium">
                        {(((selectedAgent.tasksCompleted - selectedAgent.errorsToday) / selectedAgent.tasksCompleted) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Specialty Description */}
              <div className="bg-blue-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2">Agent Specialty</h4>
                <p className="text-sm text-gray-700">{selectedAgent.specialty}</p>
              </div>

              {/* Performance Recommendations */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Performance Recommendations</h4>
                <div className="space-y-2">
                  {selectedAgent.load > 90 && (
                    <div className="flex items-start p-3 bg-red-50 rounded-lg">
                      <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 mr-3" />
                      <div>
                        <p className="font-medium text-red-800">High Load Warning</p>
                        <p className="text-sm text-red-700">Consider scaling resources or load balancing</p>
                      </div>
                    </div>
                  )}
                  {selectedAgent.accuracy < 95 && (
                    <div className="flex items-start p-3 bg-yellow-50 rounded-lg">
                      <AlertCircle className="w-5 h-5 text-yellow-500 mt-0.5 mr-3" />
                      <div>
                        <p className="font-medium text-yellow-800">Accuracy Below Target</p>
                        <p className="text-sm text-yellow-700">Consider model retraining or parameter adjustment</p>
                      </div>
                    </div>
                  )}
                  {selectedAgent.errorsToday === 0 && selectedAgent.accuracy > 97 && (
                    <div className="flex items-start p-3 bg-green-50 rounded-lg">
                      <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 mr-3" />
                      <div>
                        <p className="font-medium text-green-800">Excellent Performance</p>
                        <p className="text-sm text-green-700">Agent is performing optimally with no errors today</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-4 border-t mt-6">
              <button
                onClick={() => setSelectedAgent(null)}
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

export default PerformanceMonitoring;