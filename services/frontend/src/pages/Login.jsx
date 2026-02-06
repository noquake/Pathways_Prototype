import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth'; 
import './Login.css'; 

function Login() {
  const navigate = useNavigate();
  
  // 1. GET LOGOUT AND ROLE FROM THE HOOK
  const { isAuthenticated, userRole, login, logout, isInitialized } = useAuth();


  // 2. DEFINE THE NAVIGATION FUNCTION
  const handleDashboardNavigation = () => {
    console.log("Navigating to dashboard for role:", userRole);
    
    let target = '/dashboard';
    if (userRole === 'admin') target = '/admin-dashboard';
    else if (userRole === 'hr') target = '/hr-dashboard';
    else if (userRole === 'practitioner') target = '/practitioner-dashboard';
    
    navigate(target);
  };

  // Debug log to prove the new file is loaded
  console.log("Login Page Loaded. Auth Status:", isAuthenticated, "Role:", userRole);

  if (!isInitialized) {
      return <div className="loading-screen">Loading authentication...</div>;
  }

  return (
    <div className="login-page">
      <div className="login-container">
        
        {isAuthenticated ? (
          /* VIEW 1: USER IS LOGGED IN */
          <>
            <h2>Welcome Back!</h2>
            <p>You are currently logged in as: <strong>{userRole || 'User'}</strong></p>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <button 
                onClick={handleDashboardNavigation} 
                className="login-button"
                style={{ backgroundColor: 'var(--color-secondary, #2d5a27)' }}
              >
                Go to Dashboard
              </button>

              <button 
                onClick={logout} 
                className="login-button"
                style={{ backgroundColor: '#64748b' }}
              >
                Log Out
              </button>
            </div>
          </>
        ) : (
          /* VIEW 2: USER IS LOGGED OUT */
          <>
            <h2>Login Required</h2>
            <p>Please log in to access practitioner features, HR dashboard, or admin dashboard.</p>
            <button 
              onClick={login} 
              className="login-button"
            >
              Login with Keycloak
            </button>
          </>
        )}

      </div>
    </div>
  );
}

export default Login;