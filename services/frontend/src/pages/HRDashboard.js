import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

function HRDashboard({ apiUrl, keycloak }) {
  const [stats, setStats] = useState(null);
  const [practitioners, setPractitioners] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      // In MVP 1, these would be actual API endpoints
      // For now, we'll use placeholder data
      setStats({
        publicQueries: 0,
        practitionerQueries: 0,
        activePractitioners: 0,
        totalPathways: 0
      });
      setPractitioners([]);
    } catch (error) {
      console.error('Error loading HR data:', error);
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
        <h2>HR Dashboard</h2>
        <p className="dashboard-description">
          Read-only metadata and aggregate statistics. No access to query content.
        </p>

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

        <div className="practitioners-section">
          <h3>Registered Practitioners</h3>
          {practitioners.length === 0 ? (
            <p>No practitioners registered yet.</p>
          ) : (
            <table className="practitioners-table">
              <thead>
                <tr>
                  <th>User ID</th>
                  <th>Email</th>
                  <th>Role</th>
                  <th>Created At</th>
                  <th>Last Login</th>
                  <th>Total Queries</th>
                </tr>
              </thead>
              <tbody>
                {practitioners.map((p) => (
                  <tr key={p.user_id}>
                    <td>{p.user_id}</td>
                    <td>{p.email}</td>
                    <td>{p.role}</td>
                    <td>{new Date(p.created_at).toLocaleDateString()}</td>
                    <td>{p.last_login ? new Date(p.last_login).toLocaleDateString() : 'Never'}</td>
                    <td>{p.total_queries || 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

export default HRDashboard;

