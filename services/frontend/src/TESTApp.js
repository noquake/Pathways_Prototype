import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Keycloak from 'keycloak-js';
import Header from './components/Header';
import PublicChat from './pages/PublicChat';
import PractitionerChat from './pages/PractitionerChat';
import HRDashboard from './pages/HRDashboard';
import AdminDashboard from './pages/AdminDashboard';
import Login from './pages/Login';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const KEYCLOAK_URL = process.env.REACT_APP_KEYCLOAK_URL || 'http://localhost:8080';
const KEYCLOAK_REALM = process.env.REACT_APP_KEYCLOAK_REALM || 'pathways';
const KEYCLOAK_CLIENT_ID = process.env.REACT_APP_KEYCLOAK_CLIENT_ID || 'pathways-frontend';

function App() {
  const [keycloak, setKeycloak] = useState(null);
  const [keycloakInstance, setKeycloakInstance] = useState(null);
  const [authenticated, setAuthenticated] = useState(false);
  const [userRole, setUserRole] = useState('public');
  const [keycloakReady, setKeycloakReady] = useState(false);

  useEffect(() => {
    const kc = new Keycloak({
      url: KEYCLOAK_URL,
      realm: KEYCLOAK_REALM,
      clientId: KEYCLOAK_CLIENT_ID
    });

    setKeycloakInstance(kc);

    // Handle authentication status updates
    const updateAuthState = (kcInstance) => {
      const isAuthenticated = kcInstance.authenticated;
      setAuthenticated(isAuthenticated);
      if (isAuthenticated && kcInstance.tokenParsed) {
        const token = kcInstance.tokenParsed;
        setUserRole(token?.realm_access?.roles?.[0] || 'public');
      } else {
        setUserRole('public');
      }
    };

    // Handle token refresh
    kc.onTokenExpired = () => {
      kc.updateToken(30).then((refreshed) => {
        if (refreshed) {
          console.log('Token refreshed');
          updateAuthState(kc);
        }
      }).catch(() => {
        console.error('Failed to refresh token');
        setAuthenticated(false);
      });
    };

    // Handle successful authentication
    kc.onAuthSuccess = () => {
      console.log('Authentication successful');
      updateAuthState(kc);
    };

    // Handle authentication errors
    kc.onAuthError = (error) => {
      console.error('Authentication error:', error);
      setAuthenticated(false);
    };

    kc.init({ onLoad: 'check-sso', checkLoginIframe: false })
      .then((authenticated) => {
        setKeycloak(kc);
        setKeycloakReady(true);
        updateAuthState(kc);
        
        // If redirected back from login, update state
        if (window.location.hash || window.location.search.includes('code=')) {
          updateAuthState(kc);
        }
      })
      .catch((error) => {
        console.error('Keycloak initialization failed:', error);
        setKeycloakReady(true); // Set ready even on error so login can be attempted
        setKeycloakInstance(kc); // Ensure instance is set for login attempts
      });
  }, []);

  const handleLogin = () => {
    // Use the instance even if not fully initialized
    const kc = keycloak || keycloakInstance;
    if (kc) {
      kc.login().catch((error) => {
        console.error('Keycloak login failed:', error);
      });
    } else {
      console.error('Keycloak instance not available');
    }
  };

  const handleLogout = () => {
    const kc = keycloak || keycloakInstance;
    if (kc) {
      kc.logout().catch((error) => {
        console.error('Keycloak logout failed:', error);
      });
    }
  };

  return (
    <Router>
      <div className="App">
        <Header 
          authenticated={authenticated}
          userRole={userRole}
          onLogin={handleLogin}
          onLogout={handleLogout}
        />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<PublicChat apiUrl={API_URL} />} />
            <Route path="/login" element={<Login onLogin={handleLogin} />} />
            <Route 
              path="/practitioner" 
              element={
                authenticated && userRole === 'practitioner' ? (
                  <PractitionerChat apiUrl={API_URL} keycloak={keycloak} />
                ) : (
                  <Navigate to="/login" />
                )
              } 
            />
            <Route 
              path="/hr" 
              element={
                authenticated && (userRole === 'hr' || userRole === 'admin') ? (
                  <HRDashboard apiUrl={API_URL} keycloak={keycloak} />
                ) : (
                  <Navigate to="/login" />
                )
              } 
            />
            <Route 
              path="/admin" 
              element={
                authenticated && userRole === 'admin' ? (
                  <AdminDashboard apiUrl={API_URL} keycloak={keycloak} />
                ) : (
                  <Navigate to="/login" />
                )
              } 
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

