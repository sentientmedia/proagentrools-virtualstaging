import React from 'react';
import { useAuth } from '../../contexts/AuthContext';
import AuthModal from './AuthModal';
import AuthLoadingPage from './AuthLoadingPage';

const ProtectedRoute = ({ children, showAuthModal = false }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <AuthLoadingPage isOAuthFlow={false} />;
  }

  if (!isAuthenticated) {
    if (showAuthModal) {
      return (
        <div>
          {children}
          <AuthModal isOpen={true} onClose={() => {}} />
        </div>
      );
    }
    
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Authentication Required</h2>
          <p className="text-gray-600 mb-6">
            Please sign in to access this feature. Create an account to get started with 100 free credits!
          </p>
          <AuthModal isOpen={true} onClose={() => window.location.href = '/'} />
        </div>
      </div>
    );
  }

  return children;
};

export default ProtectedRoute;