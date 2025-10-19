import React, { useState } from 'react';
import axios from 'axios';
import { useAuth } from '../../contexts/AuthContext';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const BrokerProfileSetup = ({ onComplete, onSkip }) => {
  const { token, user, refreshUser } = useAuth();
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(1); // 1: basics, 2: brokerage, 3: social, 4: preferences
  
  const [profile, setProfile] = useState({
    // Contact
    phone: '',
    website: '',
    bio: '',
    
    // Brokerage
    brokerage_name: '',
    
    // Social
    facebook_url: '',
    instagram_url: '',
    linkedin_url: '',
    twitter_url: '',
    
    // Preferences
    default_writing_style: 'professional',
    include_contact_in_content: true
  });

  const writingStyles = [
    { value: 'data_driven_dossier', label: 'Data-Driven Dossier', description: 'Factual, comprehensive, analytical with statistics and research' },
    { value: 'storybook_sanctuary', label: 'Storybook Sanctuary', description: 'Enchanting, whimsical, creates wonder and peace' },
    { value: 'coastal_canvas', label: 'Coastal Canvas', description: 'Light, airy, seaside essence with natural beauty' },
    { value: 'urban_quip', label: 'Urban Quip', description: 'Witty, sophisticated, modern city-centric vocabulary' },
    { value: 'material_minimalist', label: 'Material Minimalist', description: 'Clean, uncluttered, focuses on textures and finishes' },
    { value: 'high_voltage_hype', label: 'High-Voltage Hype', description: 'Exciting, energetic, creates urgency and appeal' },
    { value: 'homestead_harmony', label: 'Homestead Harmony', description: 'Warm, traditional, welcoming family home feel' },
    { value: 'worldly_opulence', label: 'Worldly Opulence', description: 'Luxury, sophisticated, global influences and refined taste' },
    { value: 'roi_realtalk', label: 'ROI RealTalk', description: 'Practical, value-focused, straightforward financial emphasis' },
    { value: 'loft_luxe', label: 'Loft Luxe', description: 'Industrial meets sophisticated luxury, raw with high-end finishes' },
    { value: 'retro_modernist', label: 'Retro Modernist', description: 'Mid-century modern with contemporary twist' },
    { value: 'alpine_air', label: 'Alpine Air', description: 'Fresh, crisp, invigorating mountain environment feel' },
    { value: 'block_by_block_chronicle', label: 'Block-by-Block Chronicle', description: 'Detailed, sequential narrative of development process' },
    { value: 'skyline_sonnet', label: 'Skyline Sonnet', description: 'Poetic, grand urban imagery with contemplative beauty' },
    { value: 'straight_shooter', label: 'Straight Shooter', description: 'Direct, honest, no-nonsense gets straight to the point' }
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await axios.put(`${BACKEND_URL}/api/profile`, profile, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      await refreshUser();
      
      if (onComplete) {
        onComplete();
      }
    } catch (err) {
      console.error('Failed to save profile:', err);
      alert(err.response?.data?.detail || 'Failed to save profile');
    } finally {
      setLoading(false);
    }
  };

  const handleSkip = () => {
    if (onSkip) {
      onSkip();
    }
  };

  const renderStep1 = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome, {user?.full_name}!</h2>
        <p className="text-gray-600">Let's set up your broker profile so AI can personalize your content</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Phone Number <span className="text-red-500">*</span>
        </label>
        <input
          type="tel"
          value={profile.phone}
          onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
          placeholder="(555) 123-4567"
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Website (Optional)
        </label>
        <input
          type="url"
          value={profile.website}
          onChange={(e) => setProfile({ ...profile, website: e.target.value })}
          placeholder="https://yourwebsite.com"
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Professional Bio
        </label>
        <textarea
          value={profile.bio}
          onChange={(e) => setProfile({ ...profile, bio: e.target.value })}
          placeholder="Tell clients about your experience and expertise..."
          rows="4"
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 resize-none"
        />
        <p className="text-sm text-gray-500 mt-1">This will appear in your marketing materials</p>
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Brokerage Information</h2>
        <p className="text-gray-600">Add your brokerage details</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Brokerage Name
        </label>
        <input
          type="text"
          value={profile.brokerage_name}
          onChange={(e) => setProfile({ ...profile, brokerage_name: e.target.value })}
          placeholder="ABC Realty"
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Social Media (Optional)</h2>
        <p className="text-gray-600">Connect your social profiles</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Facebook Profile
        </label>
        <input
          type="url"
          value={profile.facebook_url}
          onChange={(e) => setProfile({ ...profile, facebook_url: e.target.value })}
          placeholder="https://facebook.com/yourprofile"
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Instagram Profile
        </label>
        <input
          type="url"
          value={profile.instagram_url}
          onChange={(e) => setProfile({ ...profile, instagram_url: e.target.value })}
          placeholder="https://instagram.com/yourprofile"
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          LinkedIn Profile
        </label>
        <input
          type="url"
          value={profile.linkedin_url}
          onChange={(e) => setProfile({ ...profile, linkedin_url: e.target.value })}
          placeholder="https://linkedin.com/in/yourprofile"
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Writing Preferences</h2>
        <p className="text-gray-600">Choose your default content style</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Default Writing Style
        </label>
        <div className="space-y-3">
          {writingStyles.map(style => (
            <div
              key={style.value}
              onClick={() => setProfile({ ...profile, default_writing_style: style.value })}
              className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
                profile.default_writing_style === style.value
                  ? 'border-blue-600 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-gray-900">{style.label}</div>
                  <div className="text-sm text-gray-600 mt-1">{style.description}</div>
                </div>
                {profile.default_writing_style === style.value && (
                  <div className="text-blue-600 text-2xl">✓</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-4">
        <label className="flex items-start space-x-3 cursor-pointer">
          <input
            type="checkbox"
            checked={profile.include_contact_in_content}
            onChange={(e) => setProfile({ ...profile, include_contact_in_content: e.target.checked })}
            className="mt-1 w-5 h-5 text-blue-600"
          />
          <div>
            <div className="font-medium text-gray-900">Include contact info in AI content</div>
            <div className="text-sm text-gray-600 mt-1">
              Automatically add your phone and email to marketing materials
            </div>
          </div>
        </label>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-6">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full p-8">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            {[1, 2, 3, 4].map(s => (
              <div
                key={s}
                className={`flex-1 h-2 rounded-full mx-1 transition-all ${
                  s <= step ? 'bg-blue-600' : 'bg-gray-200'
                }`}
              />
            ))}
          </div>
          <div className="text-sm text-gray-600 text-center">
            Step {step} of 4
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          {step === 1 && renderStep1()}
          {step === 2 && renderStep2()}
          {step === 3 && renderStep3()}
          {step === 4 && renderStep4()}

          {/* Navigation Buttons */}
          <div className="flex items-center justify-between mt-8 pt-6 border-t border-gray-200">
            <div>
              {step > 1 && (
                <button
                  type="button"
                  onClick={() => setStep(step - 1)}
                  className="text-gray-600 hover:text-gray-900 font-medium"
                >
                  ← Back
                </button>
              )}
            </div>

            <div className="flex space-x-3">
              <button
                type="button"
                onClick={handleSkip}
                className="px-6 py-3 text-gray-600 hover:text-gray-900 font-medium"
              >
                Skip for now
              </button>

              {step < 4 ? (
                <button
                  type="button"
                  onClick={() => setStep(step + 1)}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
                >
                  Next →
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={loading || !profile.phone}
                  className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-colors font-medium disabled:opacity-50"
                >
                  {loading ? 'Saving...' : 'Complete Setup 🎉'}
                </button>
              )}
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export default BrokerProfileSetup;
