import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import BrandingManager from '../branding/BrandingManager';

const UserDashboard = () => {
  const { user, logout, refreshUserData } = useAuth();
  const [copied, setCopied] = useState(false);
  const [showBrandingManager, setShowBrandingManager] = useState(false);

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Loading...</h2>
          <div className="animate-pulse bg-gray-300 h-4 w-32 rounded mx-auto"></div>
        </div>
      </div>
    );
  }

  const copyReferralCode = () => {
    navigator.clipboard.writeText(user.referral_code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleRefreshCredits = () => {
    refreshUserData();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-400 rounded-lg flex items-center justify-center">
                <span className="text-xl font-bold text-white">PA</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">ProAgentTools</h1>
                <p className="text-sm text-gray-600">Dashboard</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="text-gray-600 hover:text-gray-900 font-medium"
            >
              Sign Out
            </button>
          </div>
        </div>
      </div>

      {/* Dashboard Content */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-6">
          
          {/* User Profile Card */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Profile</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-500">Name</label>
                  <p className="text-lg text-gray-900">{user.full_name}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Email</label>
                  <p className="text-gray-900">{user.email}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Subscription</label>
                  <span className={`inline-flex px-2 py-1 text-xs rounded-full ${
                    user.subscription_status === 'active' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {user.subscription_status}
                  </span>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Member Since</label>
                  <p className="text-gray-900">{new Date(user.created_at).toLocaleDateString()}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Credits & Usage */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Credits Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900">Credits</h2>
                <button
                  onClick={handleRefreshCredits}
                  className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                >
                  Refresh
                </button>
              </div>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="text-3xl font-bold text-blue-600">{user.credits}</div>
                  <p className="text-blue-600 font-medium">Available Credits</p>
                  <p className="text-sm text-gray-600 mt-1">
                    Use for AI tools and interior designs
                  </p>
                </div>
                <div className="bg-green-50 rounded-lg p-4">
                  <div className="text-3xl font-bold text-green-600">{user.total_referrals || 0}</div>
                  <p className="text-green-600 font-medium">Referrals Made</p>
                  <p className="text-sm text-gray-600 mt-1">
                    Each successful referral earns 100 credits
                  </p>
                </div>
              </div>
            </div>

            {/* Referral Program Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Referral Program</h2>
              <div className="bg-gray-50 rounded-lg p-4 mb-4">
                <p className="text-gray-700 mb-2">
                  Share your referral code and earn <strong>100 credits</strong> for each friend who becomes a paying member!
                </p>
                <div className="flex items-center space-x-2">
                  <input
                    type="text"
                    value={user.referral_code}
                    readOnly
                    className="flex-1 px-3 py-2 bg-white border border-gray-300 rounded-lg"
                  />
                  <button
                    onClick={copyReferralCode}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      copied
                        ? 'bg-green-600 text-white'
                        : 'bg-blue-600 text-white hover:bg-blue-700'
                    }`}
                  >
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                </div>
              </div>
              <div className="text-sm text-gray-600">
                <p>• Your friend gets 100 bonus credits when they sign up with your code</p>
                <p>• You get 100 credits when they become a paying member</p>
                <p>• No limit on referrals!</p>
              </div>
            </div>

            {/* Agent Branding */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900">Agent Branding</h2>
                <button
                  onClick={() => setShowBrandingManager(true)}
                  className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                >
                  Manage
                </button>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between py-2">
                  <span className="text-gray-700">Logo Watermarks</span>
                  <span className="text-gray-600 text-sm">Auto-applied to all images</span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-gray-700">Brand Consistency</span>
                  <span className="text-gray-600 text-sm">Across all marketing materials</span>
                </div>
              </div>
              <button
                onClick={() => setShowBrandingManager(true)}
                className="w-full mt-4 py-2 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors"
              >
                Upload Logo & Configure Watermarks
              </button>
            </div>

            {/* Account Settings */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Account Settings</h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between py-2">
                  <div>
                    <span className="text-gray-700 font-medium">Credit Confirmations</span>
                    <p className="text-sm text-gray-500">Show confirmation before processing AI tools</p>
                  </div>
                  <button
                    onClick={() => {
                      const isDisabled = localStorage.getItem('creditConfirmationDisabled') === 'true';
                      if (isDisabled) {
                        localStorage.removeItem('creditConfirmationDisabled');
                        alert('Credit confirmations have been re-enabled.');
                      } else {
                        localStorage.setItem('creditConfirmationDisabled', 'true');
                        alert('Credit confirmations have been disabled.');
                      }
                    }}
                    className={`px-3 py-1 text-xs rounded-full transition-colors ${
                      localStorage.getItem('creditConfirmationDisabled') === 'true'
                        ? 'bg-gray-100 text-gray-600'
                        : 'bg-green-100 text-green-600'
                    }`}
                  >
                    {localStorage.getItem('creditConfirmationDisabled') === 'true' ? 'Disabled' : 'Enabled'}
                  </button>
                </div>
              </div>
            </div>

            {/* Tools Usage */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Tool Costs</h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between py-2 border-b border-gray-100">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="text-blue-600 text-sm">🏠</span>
                    </div>
                    <span className="font-medium text-gray-900">Interior Design AI</span>
                  </div>
                  <span className="text-gray-600 font-medium">5 credits</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-gray-100">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                      <span className="text-green-600 text-sm">✍️</span>
                    </div>
                    <span className="font-medium text-gray-900">AI Marketing Tools</span>
                  </div>
                  <span className="text-gray-600 font-medium">1-4 credits each</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Branding Manager Modal */}
      {showBrandingManager && (
        <BrandingManager onClose={() => setShowBrandingManager(false)} />
      )}
    </div>
  );
};

export default UserDashboard;