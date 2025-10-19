import React, { useState } from 'react';
import axios from 'axios';
import { useAuth } from '../../contexts/AuthContext';
import BrokerProfileSetup from '../profile/BrokerProfileSetup';
import CreateListingPage from '../listings/CreateListingPage';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const OnboardingWizard = ({ onComplete }) => {
  const { login, refreshUser } = useAuth();
  const [currentStep, setCurrentStep] = useState(1); // 1: signup, 2: profile, 3: first listing
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Step 1: Sign Up Data
  const [signupData, setSignupData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    fullName: '',
    referralCode: ''
  });

  // Step 1: Sign Up Form
  const handleSignUp = async (e) => {
    e.preventDefault();
    setError('');

    if (signupData.password !== signupData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (signupData.password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setLoading(true);

    try {
      // Register user
      const registerResponse = await axios.post(`${BACKEND_URL}/api/auth/register`, {
        email: signupData.email,
        password: signupData.password,
        full_name: signupData.fullName,
        referral_code: signupData.referralCode || undefined
      });

      // Auto-login after registration
      const loginResponse = await axios.post(`${BACKEND_URL}/api/auth/login`, {
        email: signupData.email,
        password: signupData.password
      });

      // Set auth context
      await login(loginResponse.data.access_token);
      
      // Move to profile setup
      setCurrentStep(2);
    } catch (err) {
      console.error('Registration error:', err);
      setError(err.response?.data?.detail || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleProfileComplete = async () => {
    await refreshUser();
    setCurrentStep(3);
  };

  const handleSkipProfile = () => {
    setCurrentStep(3);
  };

  const handleFirstListingCreated = () => {
    if (onComplete) {
      onComplete();
    }
  };

  const handleSkipListing = () => {
    if (onComplete) {
      onComplete();
    }
  };

  // Step 1: Sign Up Form
  if (currentStep === 1) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-indigo-900 flex items-center justify-center p-6">
        <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-block bg-blue-100 rounded-full p-4 mb-4">
              <svg className="w-12 h-12 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
            </div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome to ProAgentTools</h1>
            <p className="text-gray-600">Create your account to get started</p>
          </div>

          {/* Progress Steps */}
          <div className="flex items-center justify-center space-x-2 mb-8">
            <div className="flex items-center">
              <div className="bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold">1</div>
              <span className="ml-2 text-sm font-medium text-gray-900">Sign Up</span>
            </div>
            <div className="w-12 h-1 bg-gray-200"></div>
            <div className="flex items-center">
              <div className="bg-gray-200 text-gray-500 w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold">2</div>
              <span className="ml-2 text-sm text-gray-400">Profile</span>
            </div>
            <div className="w-12 h-1 bg-gray-200"></div>
            <div className="flex items-center">
              <div className="bg-gray-200 text-gray-500 w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold">3</div>
              <span className="ml-2 text-sm text-gray-400">Listing</span>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
              {error}
            </div>
          )}

          {/* Benefits Banner */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg p-4 mb-6">
            <div className="flex items-center space-x-2 mb-2">
              <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="font-semibold text-green-900">Start with 100 FREE credits</span>
            </div>
            <ul className="text-sm text-green-800 space-y-1">
              <li>• Create up to 5 listings</li>
              <li>• AI property descriptions</li>
              <li>• Professional marketing content</li>
            </ul>
          </div>

          <form onSubmit={handleSignUp} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Full Name *</label>
              <input
                type="text"
                value={signupData.fullName}
                onChange={(e) => setSignupData({ ...signupData, fullName: e.target.value })}
                placeholder="John Smith"
                required
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email *</label>
              <input
                type="email"
                value={signupData.email}
                onChange={(e) => setSignupData({ ...signupData, email: e.target.value })}
                placeholder="john@realestate.com"
                required
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password *</label>
              <input
                type="password"
                value={signupData.password}
                onChange={(e) => setSignupData({ ...signupData, password: e.target.value })}
                placeholder="••••••••"
                required
                minLength="8"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">Minimum 8 characters</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password *</label>
              <input
                type="password"
                value={signupData.confirmPassword}
                onChange={(e) => setSignupData({ ...signupData, confirmPassword: e.target.value })}
                placeholder="••••••••"
                required
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Referral Code (Optional)</label>
              <input
                type="text"
                value={signupData.referralCode}
                onChange={(e) => setSignupData({ ...signupData, referralCode: e.target.value })}
                placeholder="Enter code if you have one"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-lg font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-50 text-lg"
            >
              {loading ? 'Creating Account...' : 'Create Account & Continue →'}
            </button>
          </form>

          <p className="text-center text-sm text-gray-600 mt-6">
            Already have an account?{' '}
            <a href="/login" className="text-blue-600 hover:text-blue-700 font-medium">
              Sign In
            </a>
          </p>
        </div>
      </div>
    );
  }

  // Step 2: Broker Profile Setup
  if (currentStep === 2) {
    return (
      <BrokerProfileSetup 
        onComplete={handleProfileComplete}
        onSkip={handleSkipProfile}
      />
    );
  }

  // Step 3: Create First Listing
  if (currentStep === 3) {
    return (
      <div className="min-h-screen bg-gray-50">
        {/* Skip Option Banner */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4">
          <div className="max-w-4xl mx-auto px-6 flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">Almost Done! Create Your First Listing</h2>
              <p className="text-blue-100 text-sm">Or skip and explore the dashboard first</p>
            </div>
            <button
              onClick={handleSkipListing}
              className="bg-white text-blue-600 px-6 py-2 rounded-lg font-medium hover:bg-blue-50 transition-colors"
            >
              Skip for Now →
            </button>
          </div>
        </div>

        <CreateListingPage 
          onListingCreated={handleFirstListingCreated}
          onClose={handleSkipListing}
        />
      </div>
    );
  }

  return null;
};

export default OnboardingWizard;
