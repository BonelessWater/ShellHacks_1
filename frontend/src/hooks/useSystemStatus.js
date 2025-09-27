import { useState, useEffect } from 'react';

export const useSystemStatus = () => {
  const [systemStatus, setSystemStatus] = useState({
    agents: {
      orchestrator: { status: 'active', load: 85 },
      validation: { status: 'active', load: 72 },
      fraud_detection: { status: 'active', load: 93 },
      analytics: { status: 'active', load: 68 }
    }
  });

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      setSystemStatus(prev => ({
        ...prev,
        agents: Object.fromEntries(
          Object.entries(prev.agents).map(([name, agent]) => [
            name,
            {
              ...agent,
              load: Math.max(20, Math.min(100, agent.load + (Math.random() - 0.5) * 10))
            }
          ])
        )
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return { systemStatus };
};