import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import axios from 'axios';

const BrandingManager = ({ onClose }) => {
  const { token } = useAuth();
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

  const [branding, setBranding] = useState({
    logo_url: null,
    watermark_position: 'bottom-right',
    watermark_opacity: 0.7,
    brand_colors: {}
  });
  
  const [uploading, setUploading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const positions = [
    { value: 'top-left', label: 'Top Left' },
    { value: 'top-right', label: 'Top Right' },
    { value: 'bottom-left', label: 'Bottom Left' },
    { value: 'bottom-right', label: 'Bottom Right' },
    { value: 'center', label: 'Center' }
  ];

  useEffect(() => {
    loadBrandingSettings();
  }, []);

  const loadBrandingSettings = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/branding/settings`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setBranding(response.data);
    } catch (err) {
      console.error('Failed to load branding settings:', err);
    }
  };

  const handleLogoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file (PNG, JPG, GIF)');
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setError('Image must be smaller than 5MB');
      return;
    }

    setUploading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${BACKEND_URL}/api/branding/upload-logo`, formData, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });

      if (response.data.success) {
        setBranding(prev => ({
          ...prev,
          logo_url: response.data.logo_url
        }));
        setSuccess('Logo uploaded successfully!');
        setTimeout(() => setSuccess(''), 3000);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to upload logo');
    } finally {
      setUploading(false);
    }
  };

  const handleSettingsUpdate = async () => {
    setSaving(true);
    setError('');

    try {
      const response = await axios.put(
        `${BACKEND_URL}/api/branding/settings`,
        {
          position: branding.watermark_position,
          opacity: branding.watermark_opacity,
          brand_colors: branding.brand_colors
        },
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.data.success) {
        setSuccess('Settings updated successfully!');
        setTimeout(() => setSuccess(''), 3000);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to update settings');
    } finally {
      setSaving(false);
    }
  };

  const handleInputChange = (field, value) => {
    setBranding(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-900">Agent Branding & Watermarks</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          
          {/* Logo Upload Section */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent Logo</h3>
            
            {branding.logo_url && (
              <div className="mb-4">
                <img 
                  src={`${BACKEND_URL}${branding.logo_url}`}
                  alt="Current logo"
                  className="max-w-xs max-h-32 object-contain border border-gray-200 rounded p-2"
                />
                <p className="text-sm text-gray-600 mt-1">Current logo</p>
              </div>
            )}

            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <input
                type="file"
                accept="image/*"
                onChange={handleLogoUpload}
                className="hidden"
                id="logo-upload"
                disabled={uploading}
              />
              <label 
                htmlFor="logo-upload"
                className={`cursor-pointer ${uploading ? 'opacity-50' : ''}`}
              >
                <div className="text-gray-500">
                  {uploading ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-2"></div>
                      Uploading...
                    </div>
                  ) : (
                    <>
                      <span className="text-2xl mb-2 block">📁</span>
                      <span className="text-sm">Click to upload your logo</span>
                      <p className="text-xs text-gray-400 mt-1">PNG, JPG or GIF (max 5MB)</p>
                    </>
                  )}
                </div>
              </label>
            </div>
          </div>

          {/* Watermark Settings */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Watermark Settings</h3>
            
            <div className="grid md:grid-cols-2 gap-4">
              {/* Position */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Watermark Position
                </label>
                <select
                  value={branding.watermark_position}
                  onChange={(e) => handleInputChange('watermark_position', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {positions.map(pos => (
                    <option key={pos.value} value={pos.value}>
                      {pos.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Opacity */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Opacity ({Math.round(branding.watermark_opacity * 100)}%)
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.1"
                  value={branding.watermark_opacity}
                  onChange={(e) => handleInputChange('watermark_opacity', parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {/* Watermark Preview */}
          {branding.logo_url && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Watermark Preview</h3>
              <div className="relative bg-gray-100 rounded-lg p-4 h-48">
                <div className="w-full h-full bg-gradient-to-br from-blue-50 to-indigo-100 rounded relative">
                  <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                    Sample Property Image
                  </div>
                  <img 
                    src={`${BACKEND_URL}${branding.logo_url}`}
                    alt="Watermark preview"
                    className={`absolute w-12 h-12 object-contain`}
                    style={{
                      opacity: branding.watermark_opacity,
                      ...getPositionStyles(branding.watermark_position)
                    }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Information Box */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-medium text-blue-900 mb-2">Watermark Application</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Automatically applied to new interior design images</li>
              <li>• Applied to all AI-generated marketing images</li>
              <li>• Can be applied to uploaded property photos</li>
              <li>• Original images are preserved without watermarks</li>
            </ul>
          </div>

          {/* Error/Success Messages */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          {success && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <p className="text-green-800 text-sm">{success}</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t flex items-center justify-between">
          <button
            onClick={onClose}
            className="text-gray-600 hover:text-gray-800 font-medium"
          >
            Cancel
          </button>
          <button
            onClick={handleSettingsUpdate}
            disabled={saving}
            className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
              saving 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700'
            } text-white`}
          >
            {saving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>

      </div>
    </div>
  );
};

// Helper function for watermark positioning
const getPositionStyles = (position) => {
  const positions = {
    'top-left': { top: '10px', left: '10px' },
    'top-right': { top: '10px', right: '10px' },
    'bottom-left': { bottom: '10px', left: '10px' },
    'bottom-right': { bottom: '10px', right: '10px' },
    'center': { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }
  };
  
  return positions[position] || positions['bottom-right'];
};

export default BrandingManager;