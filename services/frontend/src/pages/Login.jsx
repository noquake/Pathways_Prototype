import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import keycloak from '../keycloak'; // Import the instance we created above
import './Login.css';

function Login() {
  const navigate = useNavigate();
  const [isLoginProcessing, setIsLoginProcessing] = useState(false);
  const didInit = useRef(false);

  useEffect(() => {
    // Prevent double-initialization in React 18 Strict Mode
    if (didInit.current) return;
    didInit.current = true;

    // Initialize Keycloak
    keycloak.init({ 
      onLoad: 'check-sso',
      silentCheckSsoRedirectUri: window.location.origin + '/silent-check-sso.html' 
    }).then(authenticated => {
      if (authenticated) {
        // User is already logged in, redirect them immediately
        handlePostLogin();
      }
    }).catch(console.error);
  }, []);

  const handlePostLogin = () => {
    // 1. Get User Groups/Roles from the token
    // Note: Adjust the path below depending on how you mapped groups in Keycloak 
    // (e.g. tokenParsed.groups, tokenParsed.realm_access.roles, or resource_access)
    const roles = keycloak.tokenParsed?.realm_access?.roles || [];
    const groups = keycloak.tokenParsed?.groups || []; // If you mapped 'groups' mapper

    // 2. Logic to determine landing page
    const landingPage = determineLandingPage(roles, groups);
    
    // 3. Navigate
    navigate(landingPage);
  };

  const determineLandingPage = (roles, groups) => {
    // Prioritize routing based on hierarchy or specific logic
    // Checks for both Roles (realm) or Groups (if mapped)
    if (roles.includes('admin') || groups.includes('admin-group')) {
      return '/admin-dashboard';
    } 
    if (roles.includes('hr') || groups.includes('hr-group')) {
      return '/hr-dashboard';
    } 
    if (roles.includes('practitioner') || groups.includes('practitioner-group')) {
      return '/practitioner-dashboard';
    }
    return '/dashboard'; 
  };

  const handleLogin = () => {
    setIsLoginProcessing(true);
    keycloak.login(); 
  };

  return (
    <div className="login-page">
      <div className="login-container">
        <h2>Login Required</h2>
        <p>Please log in to access practitioner features, HR dashboard, or admin dashboard.</p>
        
        <button 
          onClick={handleLogin} 
          className="login-button"
          disabled={isLoginProcessing}
        >
          {isLoginProcessing ? 'Redirecting...' : 'Login with Keycloak'}
        </button>
      </div>
    </div>
  );
}

export default Login;