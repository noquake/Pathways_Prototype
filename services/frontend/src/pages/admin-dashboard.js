import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

function AdminDashboard({ apiUrl, keycloak }) {
  const [stats, setStats] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      // Check backend health
      const healthResponse = await axios.get(`${apiUrl}/health`);
      setSystemHealth(healthResponse.data);

      // Load stats (same as HR for now)
      setStats({
        publicQueries: 0,
        practitionerQueries: 0,
        activePractitioners: 0,
        totalPathways: 0
      });
    } catch (error) {
      console.error('Error loading admin data:', error);
      setSystemHealth({ status: 'unhealthy' });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="dashboard-loading">Loading...</div>;
  }

  return (
    <div className="dashboard-page">
      <div className="dashboard-container">
        <h2>Admin Dashboard</h2>
        <p className="dashboard-description">
          System governance and oversight. Full access to metadata and system state.
        </p>

        <div className="system-health">
          <h3>System Health</h3>
          <div className={`health-status ${systemHealth?.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
            {systemHealth?.status || 'Unknown'}
          </div>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <h3>Public Queries</h3>
            <p className="stat-value">{stats?.publicQueries || 0}</p>
          </div>
          <div className="stat-card">
            <h3>Practitioner Queries</h3>
            <p className="stat-value">{stats?.practitionerQueries || 0}</p>
          </div>
          <div className="stat-card">
            <h3>Active Practitioners</h3>
            <p className="stat-value">{stats?.activePractitioners || 0}</p>
          </div>
          <div className="stat-card">
            <h3>Total Pathways</h3>
            <p className="stat-value">{stats?.totalPathways || 0}</p>
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

