import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth'; 
import './Login.css'; // Ensure path is correct based on where you put the CSS file

function Login() {
  const navigate = useNavigate();
  const { isAuthenticated, userRole, login, isInitialized } = useAuth();

  useEffect(() => {
    if (isInitialized && isAuthenticated) {
        let target = '/dashboard';
        
        if (userRole === 'admin') target = '/admin-dashboard';
        else if (userRole === 'hr') target = '/hr-dashboard';
        else if (userRole === 'practitioner') target = '/practitioner-dashboard';
        
        navigate(target);
    }
  }, [isInitialized, isAuthenticated, userRole, navigate]);

  // Uses the new .loading-screen class we added to CSS
  if (!isInitialized) {
      return <div className="loading-screen">Loading authentication...</div>;
  }

  return (
    // Uses .login-page from CSS
    <div className="login-page">
      
      {/* Uses .login-container from CSS */}
      <div className="login-container">
        
        {/* Matches .login-container h2 */}
        <h2>Login Required</h2>
        
        {/* Matches .login-container p */}
        <p>Please log in to access practitioner features, HR dashboard, or admin dashboard.</p>
        
        {/* Uses .login-button from CSS */}
        <button 
          onClick={login} 
          className="login-button"
        >
          Login with Keycloak
        </button>
      </div>
    </div>
  );
}

export default Login;