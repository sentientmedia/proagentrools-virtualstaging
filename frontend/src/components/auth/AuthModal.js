import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';

const AuthModal = ({ isOpen, onClose, initialMode = 'login' }) => {
  const [mode, setMode] = useState(initialMode); // 'login' or 'register'
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    fullName: '',
    referralCode: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login, register, loginWithGoogle } = useAuth();

  if (!isOpen) return null;

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user types
    if (error) setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      let result;
      
      if (mode === 'login') {
        result = await login(formData.email, formData.password);
      } else {
        if (!formData.fullName.trim()) {
          setError('Full name is required');
          setLoading(false);
          return;
        }
        result = await register(
          formData.email, 
          formData.password, 
          formData.fullName,
          formData.referralCode
        );
      }

      if (result.success) {
        onClose();
        // Reset form
        setFormData({
          email: '',
          password: '',
          fullName: '',
          referralCode: ''
        });
      } else {
        setError(result.error);
      }
    } catch (error) {
      setError('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const switchMode = () => {
    setMode(mode === 'login' ? 'register' : 'login');
    setError('');
    setFormData({
      email: '',
      password: '',
      fullName: '',
      referralCode: ''
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-900">
            {mode === 'login' ? 'Sign In' : 'Create Account'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Google OAuth Button */}
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              console.log('Google OAuth button clicked');
              loginWithGoogle();
            }}
            type="button"
            className="w-full flex items-center justify-center px-4 py-3 border border-gray-300 rounded-lg bg-white hover:bg-gray-50 transition-colors mb-4 font-medium text-gray-700"
          >
            <svg className="w-5 h-5 mr-3 flex-shrink-0" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            <span>Continue with Google</span>
          </button>

          <div className="relative flex items-center justify-center mb-4">
            <div className="border-t border-gray-300 w-full"></div>
            <span className="bg-white px-4 text-sm text-gray-500">or</span>
            <div className="border-t border-gray-300 w-full"></div>
          </div>

          {/* Help text for Google OAuth */}
          <div className="mb-4 p-2 bg-blue-50 rounded text-xs text-blue-700">
            <strong>Note:</strong> If Google login opens a page asking to "Open in new tab", click that button to continue the login process.
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === 'register' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Full Name
                </label>
                <input
                  type="text"
                  name="fullName"
                  value={formData.fullName}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter your full name"
                  required
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your email"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your password"
                required
                minLength="6"
              />
            </div>

            {mode === 'register' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Referral Code <span className="text-gray-400">(Optional)</span>
                </label>
                <input
                  type="text"
                  name="referralCode"
                  value={formData.referralCode}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter referral code for bonus credits"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Get 100 bonus credits with a valid referral code
                </p>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-red-800 text-sm">{error}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className={`w-full py-3 px-4 rounded-lg font-semibold transition-colors ${
                loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700'
              } text-white`}
            >
              {loading ? 'Processing...' : (mode === 'login' ? 'Sign In' : 'Create Account')}
            </button>
          </form>

          {/* Switch Mode */}
          <div className="text-center mt-4 pt-4 border-t border-gray-200">
            <p className="text-gray-600">
              {mode === 'login' ? "Don't have an account? " : "Already have an account? "}
              <button
                onClick={switchMode}
                className="text-blue-600 hover:text-blue-700 font-medium"
              >
                {mode === 'login' ? 'Sign up' : 'Sign in'}
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthModal;