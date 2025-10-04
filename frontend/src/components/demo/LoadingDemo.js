import React from 'react';
import AuthLoadingPage from '../auth/AuthLoadingPage';

const LoadingDemo = () => {
  return (
    <div className="min-h-screen">
      <div className="bg-gray-100 p-4 text-center">
        <h1 className="text-2xl font-bold mb-4">Authentication Loading Page Designs</h1>
        <p className="text-gray-600 mb-8">Beautiful interstitial pages for OAuth and regular loading</p>
      </div>
      
      <div className="grid md:grid-cols-2 h-screen">
        {/* OAuth Flow Loading */}
        <div className="border-r border-gray-200">
          <div className="p-4 bg-blue-50 text-center">
            <h2 className="font-semibold">Google OAuth Flow</h2>
            <p className="text-sm text-gray-600">Shows when processing OAuth session</p>
          </div>
          <AuthLoadingPage isOAuthFlow={true} />
        </div>
        
        {/* Regular Loading */}
        <div>
          <div className="p-4 bg-green-50 text-center">
            <h2 className="font-semibold">Regular Loading</h2>
            <p className="text-sm text-gray-600">Shows during normal app loading</p>
          </div>
          <AuthLoadingPage isOAuthFlow={false} />
        </div>
      </div>
    </div>
  );
};

export default LoadingDemo;