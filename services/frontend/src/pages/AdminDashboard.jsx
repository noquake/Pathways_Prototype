import { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

function AdminDashboard({ apiUrl }) {
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${apiUrl}/health`)
      .then(res => setSystemHealth(res.data))
      .catch(() => setSystemHealth({ status: 'unhealthy' }))
      .finally(() => setLoading(false));
  }, [apiUrl]);

  if (loading) return <div className="dashboard-loading">Loading...</div>;

  return (
    <div className="dashboard-page">
      <div className="dashboard-container">
        <h2>Admin Dashboard</h2>
        <p className="dashboard-description">
          System governance and oversight.
        </p>

        <div className="system-health">
          <h3>System Health</h3>
          <div className={`health-status ${systemHealth?.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
            {systemHealth?.status || 'Unknown'}
          </div>
        </div>

        <div className="admin-actions">
          <h3>Admin Actions</h3>
          <p className="coming-soon">Additional admin controls will be available in future releases.</p>
          <div className="action-buttons">
            <button className="action-button" disabled>Disable Account</button>
            <button className="action-button" disabled>Adjust Access Policies</button>
            <button className="action-button" disabled>Trigger Re-ingestion</button>
            <button className="action-button" disabled>Export Reports</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AdminDashboard;
