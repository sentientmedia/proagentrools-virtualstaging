import React, { useState, useEffect } from 'react';

const AuthLoadingPage = ({ isOAuthFlow = false }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [dots, setDots] = useState('');

  // Animated loading steps for OAuth
  const oauthSteps = [
    { step: 1, text: 'Verifying your Google account', icon: '🔐' },
    { step: 2, text: 'Setting up your profile', icon: '👤' },
    { step: 3, text: 'Preparing your workspace', icon: '🏠' }
  ];

  // Regular loading steps
  const regularSteps = [
    { step: 1, text: 'Loading your workspace', icon: '⚡' },
    { step: 2, text: 'Preparing AI tools', icon: '🤖' },
    { step: 3, text: 'Almost ready', icon: '✨' }
  ];

  const steps = isOAuthFlow ? oauthSteps : regularSteps;

  // Animate through steps
  useEffect(() => {
    if (isOAuthFlow) {
      const stepInterval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= steps.length) return prev;
          return prev + 1;
        });
      }, 1500);

      return () => clearInterval(stepInterval);
    }
  }, [isOAuthFlow, steps.length]);

  // Animate dots
  useEffect(() => {
    const dotInterval = setInterval(() => {
      setDots(prev => {
        if (prev === '...') return '';
        return prev + '.';
      });
    }, 500);

    return () => clearInterval(dotInterval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 flex items-center justify-center relative overflow-hidden">
      {/* Background Animation */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-20 w-32 h-32 bg-blue-200 rounded-full opacity-20 animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-48 h-48 bg-indigo-200 rounded-full opacity-20 animate-pulse animation-delay-1000"></div>
        <div className="absolute top-1/2 left-1/4 w-24 h-24 bg-purple-200 rounded-full opacity-20 animate-pulse animation-delay-2000"></div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 text-center max-w-md mx-auto px-6">
        {/* Logo */}
        <div className="mb-8">
          <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
            <span className="text-2xl font-bold text-white">PA</span>
          </div>
          <h1 className="text-2xl font-bold text-gray-900">ProAgentTools</h1>
        </div>

        {/* Loading Animation */}
        <div className="mb-8">
          {isOAuthFlow ? (
            // OAuth Progress Steps
            <div className="space-y-6">
              {steps.map((stepData, index) => (
                <div 
                  key={stepData.step}
                  className={`flex items-center justify-center space-x-4 transition-all duration-500 ${
                    currentStep >= stepData.step 
                      ? 'opacity-100 scale-100' 
                      : 'opacity-40 scale-95'
                  }`}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm transition-colors duration-500 ${
                    currentStep >= stepData.step 
                      ? 'bg-green-100 text-green-600' 
                      : 'bg-gray-100 text-gray-400'
                  }`}>
                    {currentStep > stepData.step ? '✓' : stepData.icon}
                  </div>
                  <span className={`text-lg transition-colors duration-500 ${
                    currentStep >= stepData.step ? 'text-gray-900' : 'text-gray-500'
                  }`}>
                    {stepData.text}
                    {currentStep === stepData.step && dots}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            // Regular Spinner
            <div className="relative">
              <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto"></div>
              <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-r-indigo-400 rounded-full animate-ping mx-auto opacity-20"></div>
            </div>
          )}
        </div>

        {/* Status Message */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            {isOAuthFlow ? 'Completing your sign in' : 'Loading your workspace'}
          </h2>
          <p className="text-gray-600 text-sm">
            {isOAuthFlow 
              ? "We're setting up your account with 100 free credits and your personal referral code!"
              : "Please wait while we prepare your AI-powered real estate tools."
            }
          </p>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2 mb-6">
          <div 
            className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full transition-all duration-1000 ease-out"
            style={{ 
              width: isOAuthFlow 
                ? `${(currentStep / steps.length) * 100}%` 
                : '60%' 
            }}
          ></div>
        </div>

        {/* Success Indicator */}
        {isOAuthFlow && currentStep > steps.length && (
          <div className="animate-fadeIn">
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">🎉</span>
            </div>
            <p className="text-green-600 font-medium">Success! Redirecting to your dashboard...</p>
          </div>
        )}

        {/* Security Notice */}
        {isOAuthFlow && (
          <div className="mt-8 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-center justify-center space-x-2 text-sm text-blue-700">
              <span>🔒</span>
              <span>Your data is secure and encrypted</span>
            </div>
          </div>
        )}
      </div>

      {/* Custom Animation Classes */}
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
        .animation-delay-1000 {
          animation-delay: 1s;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
      `}</style>
    </div>
  );
};

export default AuthLoadingPage;