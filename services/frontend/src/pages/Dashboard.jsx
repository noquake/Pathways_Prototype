import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth'; 
import './Dashboard.css'; 

const Dashboard = () => {
    const { userRole, isAuthenticated } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        // 1. Security Check: If not logged in, go to Login
        if (!isAuthenticated) {
            navigate('/login');
            return;
        }

        // 2. The Dispatcher Logic
        console.log("Dashboard Dispatcher: User Role is", userRole);
        
        switch (userRole) {
            case 'admin':
                navigate('/admin-dashboard');
                break;
            case 'hr':
                navigate('/hr-dashboard');
                break;
            case 'practitioner':
                navigate('/practitioner-dashboard');
                break;
            default:
                // Fallback: If role is missing or 'user', send Home or to a Profile page
                navigate('/'); 
        }
    }, [userRole, isAuthenticated, navigate]);

    // 3. Render the spinner while we calculate the route
    return (
        <div className="dashboard-page">
            <div className="dashboard-loading">
                <h3>Redirecting...</h3>
                <p>Please wait while we access your dashboard.</p>
            </div>
        </div>
    );
};

export default Dashboard;