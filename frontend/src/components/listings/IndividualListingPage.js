import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import axios from 'axios';

const IndividualListingPage = ({ listingId, onBack }) => {
  const { user, token } = useAuth();
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  
  const [listing, setListing] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeModule, setActiveModule] = useState('details');
  const [moduleContent, setModuleContent] = useState({});
  const [generatingModule, setGeneratingModule] = useState(null);
  const [editingModule, setEditingModule] = useState(null);
  const [editContent, setEditContent] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  
  // Image management state
  const [images, setImages] = useState([]);
  const [uploadingImages, setUploadingImages] = useState(false);
  const [selectedImageIds, setSelectedImageIds] = useState([]);
  
  // Interior design state
  const [processingDesign, setProcessingDesign] = useState(false);
  const [designSettings, setDesignSettings] = useState({
    room_type: 'living_room',
    designer: 'alessia_duval',
    color_scheme: 'glacial_muse'
  });

  const modules = [
    { id: 'details', name: 'Listing Details', icon: '📝' },
    { id: 'listing_copy', name: 'AI Copy', icon: '✍️', ai: true },
    { id: 'marketing_copy', name: 'Marketing', icon: '📢', ai: true },
    { id: 'social_media', name: 'Social Media', icon: '📱', ai: true },
    { id: 'email_template', name: 'Email Template', icon: '✉️', ai: true },
    { id: 'market_intel', name: 'Market Intel', icon: '📊', ai: true },
    { id: 'images', name: 'Images', icon: '📸' },
    { id: 'interior_design', name: 'Interior Design', icon: '🎨' },
  ];

  useEffect(() => {
    if (listingId) {
      loadListing();
      loadImages();
    }
  }, [listingId]);

  const loadListing = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/listings/${listingId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setListing(response.data);
      setModuleContent(response.data.module_outputs || {});
    } catch (err) {
      console.error('Failed to load listing:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadImages = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/listings/${listingId}/images`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setImages(response.data.photos || []);
    } catch (err) {
      console.error('Failed to load images:', err);
    }
  };

  const handleGenerateContent = async (moduleName) => {
    try {
      setGeneratingModule(moduleName);
      
      const response = await axios.post(
        `${BACKEND_URL}/api/listings/${listingId}/modules/${moduleName}/generate`,
        { module_name: moduleName },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );

      if (response.data.success) {
        setModuleContent(prev => ({
          ...prev,
          [moduleName]: {
            content: response.data.content,
            generated_at: new Date().toISOString(),
            is_ai_generated: true
          }
        }));
      }
    } catch (err) {
      console.error('Failed to generate content:', err);
      alert(err.response?.data?.detail || 'Failed to generate content');
    } finally {
      setGeneratingModule(null);
    }
  };

  const handleUpdateContent = async (moduleName) => {
    try {
      const response = await axios.put(
        `${BACKEND_URL}/api/listings/${listingId}/modules/${moduleName}`,
        { content: editContent },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );

      if (response.data.success) {
        setModuleContent(prev => ({
          ...prev,
          [moduleName]: {
            ...prev[moduleName],
            content: editContent,
            is_ai_generated: false
          }
        }));
        setEditingModule(null);
      }
    } catch (err) {
      console.error('Failed to update content:', err);
      alert('Failed to update content');
    }
  };

  const handleChatImprove = async (moduleName) => {
    if (!chatInput.trim()) return;

    try {
      setChatLoading(true);
      
      const response = await axios.post(
        `${BACKEND_URL}/api/listings/${listingId}/modules/${moduleName}/chat`,
        {
          message: chatInput,
          module_name: moduleName
        },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );

      if (response.data.success) {
        // Update the module content with AI suggestion
        alert(`AI Suggestion: ${response.data.response}\n\nYou can manually edit the content to apply changes.`);
        setChatInput('');
      }
    } catch (err) {
      console.error('Failed to chat:', err);
      alert(err.response?.data?.detail || 'Failed to get AI response');
    } finally {
      setChatLoading(false);
    }
  };

  const handleImageUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    try {
      setUploadingImages(true);
      
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const response = await axios.post(
        `${BACKEND_URL}/api/listings/${listingId}/images/upload`,
        formData,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'multipart/form-data'
          }
        }
      );

      if (response.data.success) {
        await loadImages();
        alert(`Successfully uploaded ${response.data.uploaded_count} image(s)`);
      }
    } catch (err) {
      console.error('Failed to upload images:', err);
      alert(err.response?.data?.detail || 'Failed to upload images');
    } finally {
      setUploadingImages(false);
    }
  };

  const handleDeleteImage = async (imageId) => {
    if (!window.confirm('Delete this image?')) return;

    try {
      await axios.delete(
        `${BACKEND_URL}/api/listings/${listingId}/images/${imageId}`,
        { headers: { 'Authorization': `Bearer ${token}` } }
      );
      await loadImages();
    } catch (err) {
      console.error('Failed to delete image:', err);
      alert('Failed to delete image');
    }
  };

  const handleProcessInteriorDesign = async () => {
    if (selectedImageIds.length === 0) {
      alert('Please select at least one image to process');
      return;
    }

    const creditsNeeded = selectedImageIds.length * 5;
    if (!window.confirm(`Process ${selectedImageIds.length} image(s) with interior design? This will cost ${creditsNeeded} credits.`)) {
      return;
    }

    try {
      setProcessingDesign(true);

      const response = await axios.post(
        `${BACKEND_URL}/api/listings/${listingId}/interior-design/process`,
        {
          image_ids: selectedImageIds,
          ...designSettings
        },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );

      if (response.data.success) {
        alert(`Successfully processed ${response.data.processed_count} image(s)!`);
        setSelectedImageIds([]);
        await loadListing();
      }
    } catch (err) {
      console.error('Failed to process interior design:', err);
      alert(err.response?.data?.detail || 'Failed to process images');
    } finally {
      setProcessingDesign(false);
    }
  };

  const renderModuleContent = (module) => {
    const content = moduleContent[module.id];

    if (!content) {
      if (module.ai) {
        return (
          <div className="text-center py-12">
            <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-4xl">{module.icon}</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Generate {module.name}</h3>
            <p className="text-gray-600 mb-6">Use AI to generate professional content for this module</p>
            <button
              onClick={() => handleGenerateContent(module.id)}
              disabled={generatingModule === module.id}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors disabled:bg-gray-400"
            >
              {generatingModule === module.id ? 'Generating...' : `Generate Content (1 credit)`}
            </button>
          </div>
        );
      }
      return <div className="text-gray-500 py-8 text-center">No content yet</div>;
    }

    return (
      <div className="space-y-4">
        {/* Content Display/Edit */}
        {editingModule === module.id ? (
          <div>
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              className="w-full h-64 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <div className="flex items-center space-x-2 mt-2">
              <button
                onClick={() => handleUpdateContent(module.id)}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
              >
                Save Changes
              </button>
              <button
                onClick={() => setEditingModule(null)}
                className="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div>
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="prose max-w-none whitespace-pre-wrap">{content.content}</div>
            </div>
            <div className="flex items-center justify-between mt-2">
              <div className="text-xs text-gray-500">
                {content.is_ai_generated ? '🤖 AI Generated' : '✏️ Manually Edited'} • 
                Last updated: {new Date(content.last_edited || content.generated_at).toLocaleString()}
              </div>
              <button
                onClick={() => {
                  setEditingModule(module.id);
                  setEditContent(content.content);
                }}
                className="text-sm text-blue-600 hover:text-blue-700 font-medium"
              >
                Edit Content
              </button>
            </div>
          </div>
        )}

        {/* Chat to Improve (only for AI modules) */}
        {module.ai && !editingModule && (
          <div className="border-t border-gray-200 pt-4 mt-6">
            <h4 className="text-sm font-medium text-gray-900 mb-2">💬 Chat to Improve (1 credit per message)</h4>
            <div className="flex items-start space-x-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleChatImprove(module.id)}
                placeholder="Ask AI to refine the content..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={chatLoading}
              />
              <button
                onClick={() => handleChatImprove(module.id)}
                disabled={chatLoading || !chatInput.trim()}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400"
              >
                {chatLoading ? '...' : 'Send'}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Example: "Make it more engaging" or "Add emphasis on the backyard"
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderDetails = () => {
    if (!listing) return null;
    const { property_details } = listing;

    return (
      <div className="space-y-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Property Information</h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-700">Address</label>
              <div className="text-gray-900">{property_details.address}</div>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700">City, State ZIP</label>
              <div className="text-gray-900">
                {property_details.city}, {property_details.state} {property_details.zip_code}
              </div>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700">Property Type</label>
              <div className="text-gray-900">{property_details.property_type}</div>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700">Beds / Baths</label>
              <div className="text-gray-900">{property_details.beds} bed / {property_details.baths} bath</div>
            </div>
            {property_details.sqft && (
              <div>
                <label className="text-sm font-medium text-gray-700">Square Feet</label>
                <div className="text-gray-900">{property_details.sqft.toLocaleString()} sq ft</div>
              </div>
            )}
            {property_details.listing_price && (
              <div>
                <label className="text-sm font-medium text-gray-700">Listing Price</label>
                <div className="text-gray-900 text-xl font-bold text-blue-600">
                  ${property_details.listing_price.toLocaleString()}
                </div>
              </div>
            )}
          </div>
        </div>

        {listing.description && (
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Description</h3>
            <p className="text-gray-700">{listing.description}</p>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!listing) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">Listing not found</h2>
          <button onClick={onBack} className="text-blue-600 hover:text-blue-700">
            ← Back to Listings
          </button>
        </div>
      </div>
    );
  }

  const currentModule = modules.find(m => m.id === activeModule);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={onBack}
                className="text-gray-600 hover:text-gray-900"
              >
                ← Back
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  {listing.property_details.address}
                </h1>
                <p className="text-gray-600">
                  {listing.property_details.city}, {listing.property_details.state}
                </p>
              </div>
            </div>
            <div className="text-sm">
              <div className="text-gray-600">Available Credits</div>
              <div className="text-2xl font-bold text-blue-600">{user?.credits || 0}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Module Navigation */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex space-x-1 overflow-x-auto">
            {modules.map(module => (
              <button
                key={module.id}
                onClick={() => setActiveModule(module.id)}
                className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeModule === module.id
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-600 hover:text-gray-900'
                }`}
              >
                <span className="mr-1">{module.icon}</span>
                {module.name}
                {module.ai && moduleContent[module.id] && (
                  <span className="ml-1 text-green-600">✓</span>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Module Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            {currentModule?.icon} {currentModule?.name}
          </h2>
          
          {activeModule === 'details' && renderDetails()}
          {activeModule === 'images' && (
            <div className="text-center py-12 text-gray-500">
              Image upload and management coming soon
            </div>
          )}
          {activeModule === 'interior_design' && (
            <div className="text-center py-12 text-gray-500">
              Interior design processing coming soon
            </div>
          )}
          {currentModule?.ai && renderModuleContent(currentModule)}
        </div>
      </div>
    </div>
  );
};

export default IndividualListingPage;
