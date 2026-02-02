import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

function Header() {
  const { isAuthenticated, logout, userRole } = useAuth();
  const navigate = useNavigate();
  const getDashboardLink = () => {
    // You can use the helper flags here too if you prefer
    if (userRole === 'admin') return '/admin-dashboard';
    if (userRole === 'hr') return '/hr-dashboard';
    if (userRole === 'practitioner') return '/practitioner-dashboard';
    return '/dashboard';
  };
  
  const styles = {
    navGroup: {
      display: 'flex',
      gap: '10px',
      alignItems: 'center'
    },
    dashboardButton: {
      backgroundColor: 'rgba(255,255,255,0.2)',
      color: 'white',
      border: 'none',
      padding: '5px 12px',
      borderRadius: '4px',
      cursor: 'pointer',
      fontSize: '0.85rem',
      fontWeight: '500'
    },
    header: {
      backgroundColor: 'var(--accent-burgundy, #800020)', // Matches your theme
      color: 'white',
      padding: '8px 20px', // Slimmer padding for a less invasive feel
      boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      height: '50px' // Fixed slim height
    },
    logoLink: {
      textDecoration: 'none',
      color: 'white',
      display: 'flex',
      alignItems: 'center'
    },
    logoText: {
      fontSize: '1.1rem', // Smaller, professional font size
      margin: 0,
      fontWeight: '600'
    },
    authButton: {
      backgroundColor: 'transparent',
      color: 'white',
      border: '1px solid rgba(255,255,255,0.5)', // Subtle outlined style
      padding: '5px 12px',
      borderRadius: '4px',
      cursor: 'pointer',
      fontSize: '0.85rem',
      transition: 'all 0.2s ease',
      fontWeight: '500'
    }
  };

  return (
    <header style={styles.header}>
      <Link to="/" style={styles.logoLink}>
        <h1 style={styles.logoText}>Pathways</h1>
      </Link>

      <nav style={styles.navGroup}>
        {isAuthenticated ? (
          <>
            <button 
              style={styles.dashboardButton}
              onClick={() => navigate(getDashboardLink())}
            >
              Dashboard
            </button>
          <button 
            onClick={logout} 
            style={styles.authButton}
            onMouseOver={(e) => e.target.style.backgroundColor = 'rgba(255,255,255,0.1)'}
            onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
          >
            Logout
          </button>
          </>
        ) : (
          <button 
            onClick={() => navigate('/login')} 
            style={styles.authButton}
          >
            Login
          </button>
        )}
      </nav>
    </header>
  );
}

export default Header;