import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import axios from 'axios';

const IndividualListingPage = ({ listingId, onBack }) => {
  const { user, token } = useAuth();
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  
  const [listing, setListing] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState('overview'); // 'overview' or 'module'
  const [activeModule, setActiveModule] = useState(null);
  const [moduleContent, setModuleContent] = useState({});
  const [generatingModule, setGeneratingModule] = useState(null);
  const [editingModule, setEditingModule] = useState(null);
  const [editContent, setEditContent] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  
  // Image management state
  const [images, setImages] = useState([]);
  const [uploadingImages, setUploadingImages] = useState(false);
  const [selectedImages, setSelectedImages] = useState([]); // Array of {image_id, room_type, designer, color_scheme}
  
  // Interior design state
  const [processingDesign, setProcessingDesign] = useState(false);

  const modules = [
    // Foundation modules (auto-generated, always unlocked)
    { id: 'neighborhood_research', name: 'Neighborhood Research', icon: '🏘️', ai: true, credits: 0, description: 'Detailed neighborhood analysis', isFoundation: true, category: 'Foundation' },
    { id: 'listing_copy', name: 'Property Description', icon: '✍️', ai: true, credits: 0, description: 'Professional listing copy', isFoundation: true, category: 'Foundation' },
    { id: 'market_intel', name: 'Market Intelligence', icon: '📊', ai: true, credits: 0, description: 'Strategic market positioning', isFoundation: true, category: 'Foundation' },
    
    // Marketing & Content
    { id: 'marketing_copy', name: 'Marketing Materials', icon: '📢', ai: true, credits: 1, description: 'Brochures and flyers', requiresFoundation: true, category: 'Marketing' },
    { id: 'social_media', name: 'Social Media Posts', icon: '📱', ai: true, credits: 1, description: 'Facebook, Instagram posts', requiresFoundation: true, category: 'Marketing' },
    { id: 'email_template', name: 'Email Campaign', icon: '✉️', ai: true, credits: 1, description: 'Email to potential buyers', requiresFoundation: true, category: 'Marketing' },
    { id: 'virtual_tour_script', name: 'Virtual Tour Script', icon: '🎥', ai: true, credits: 1, description: 'Video walkthrough script', requiresFoundation: true, category: 'Marketing' },
    { id: 'open_house_promo', name: 'Open House Promotion', icon: '🏠', ai: true, credits: 1, description: 'Open house marketing', requiresFoundation: true, category: 'Marketing' },
    { id: 'property_highlights', name: 'Property Highlights', icon: '⭐', ai: true, credits: 1, description: 'Key features summary', requiresFoundation: true, category: 'Marketing' },
    
    // Analysis & Strategy
    { id: 'buyer_profile', name: 'Target Buyer Profile', icon: '👥', ai: true, credits: 1, description: 'Ideal buyer demographics', requiresFoundation: true, category: 'Strategy' },
    { id: 'price_justification', name: 'Price Justification', icon: '💰', ai: true, credits: 1, description: 'Why this price works', requiresFoundation: true, category: 'Strategy' },
    { id: 'competitor_comparison', name: 'Competitor Analysis', icon: '🔍', ai: true, credits: 1, description: 'Vs nearby properties', requiresFoundation: true, category: 'Strategy' },
    { id: 'agent_talking_points', name: 'Agent Talking Points', icon: '💼', ai: true, credits: 1, description: 'Key points for showings', requiresFoundation: true, category: 'Strategy' },
    { id: 'objection_handling', name: 'Objection Handlers', icon: '🛡️', ai: true, credits: 1, description: 'Overcome buyer concerns', requiresFoundation: true, category: 'Strategy' },
    { id: 'negotiation_tips', name: 'Negotiation Strategy', icon: '🤝', ai: true, credits: 1, description: 'Pricing and offer tactics', requiresFoundation: true, category: 'Strategy' },
    
    // Client Communications  
    { id: 'seller_updates', name: 'Seller Updates', icon: '📝', ai: true, credits: 1, description: 'Weekly seller reports', requiresFoundation: true, category: 'Communications' },
    { id: 'buyer_followup', name: 'Buyer Follow-up', icon: '📞', ai: true, credits: 1, description: 'Post-showing messages', requiresFoundation: true, category: 'Communications' },
    
    // Media
    { id: 'images', name: 'Photos & Interior Design', icon: '📸', description: 'Upload and enhance images', category: 'Media' },
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

  const [chatSuggestion, setChatSuggestion] = useState(null); // Store AI suggestion

  const handleDownloadContent = (moduleName, content) => {
    const element = document.createElement('a');
    const file = new Blob([content.replace(/\*\*/g, '')], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `${listing.property_details.address}_${moduleName}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
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
        setChatSuggestion(response.data.response);
        setChatInput('');
      }
    } catch (err) {
      console.error('Failed to chat:', err);
      alert(err.response?.data?.detail || 'Failed to get AI response');
    } finally {
      setChatLoading(false);
    }
  };

  const handleReplaceSuggestion = async (moduleName) => {
    try {
      await axios.put(
        `${BACKEND_URL}/api/listings/${listingId}/modules/${moduleName}`,
        { content: chatSuggestion },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );
      
      setModuleContent(prev => ({
        ...prev,
        [moduleName]: {
          ...prev[moduleName],
          content: chatSuggestion,
          is_ai_generated: false
        }
      }));
      setChatSuggestion(null);
      await loadListing();
    } catch (err) {
      console.error('Failed to replace:', err);
    }
  };

  const handleRejectSuggestion = () => {
    setChatSuggestion(null);
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
    if (selectedImages.length === 0) {
      alert('Please select at least one image to process');
      return;
    }

    const creditsNeeded = selectedImages.length * 5;
    if (!window.confirm(`Process ${selectedImages.length} image(s) with interior design? This will cost ${creditsNeeded} credits.`)) {
      return;
    }

    try {
      setProcessingDesign(true);

      const response = await axios.post(
        `${BACKEND_URL}/api/listings/${listingId}/interior-design/process`,
        {
          images: selectedImages  // Send per-image settings
        },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );

      if (response.data.success) {
        alert(`${response.data.message}\n\nProcessed: ${response.data.processed_count} image(s)`);
        setSelectedImages([]);
        await loadListing();
        await loadImages();
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
              {generatingModule === module.id ? 'Generating...' : `Generate Content (${module.credits || 1} credit${module.credits > 1 ? 's' : ''})`}
            </button>
          </div>
        );
      }
      return <div className="text-gray-500 py-8 text-center">No content yet</div>;
    }

    return (
      <div className="space-y-6">
        {/* Content Display/Edit */}
        {editingModule === module.id ? (
          <div>
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              className="w-full h-64 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
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
              <div className="max-w-none whitespace-pre-wrap text-gray-900 leading-relaxed">
                {content.content.replace(/\*\*/g, '')}
              </div>
            </div>
            <div className="flex items-center justify-between mt-2">
              <div className="text-xs text-gray-500">
                {content.is_ai_generated ? '🤖 AI Generated' : '✏️ Manually Edited'} • 
                Last updated: {new Date(content.last_edited || content.generated_at).toLocaleString()}
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => handleDownloadContent(module.id, content.content)}
                  className="text-sm text-green-600 hover:text-green-700 font-medium"
                >
                  Download
                </button>
                <button
                  onClick={() => {
                    setEditingModule(module.id);
                    setEditContent(content.content);
                  }}
                  className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleGenerateContent(module.id)}
                  disabled={generatingModule === module.id}
                  className="text-sm text-purple-600 hover:text-purple-700 font-medium disabled:text-gray-400"
                >
                  Regenerate
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Inline Chat Interface */}
        {module.ai && !editingModule && (
          <div className="border-t border-gray-200 pt-6">
            <h4 className="text-base font-semibold text-gray-900 mb-4">💬 Chat to Improve (1 credit per message)</h4>
            
            {/* Chat Input */}
            <div className="flex items-start space-x-2 mb-4">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleChatImprove(module.id)}
                placeholder="e.g., Make it more engaging, Add emphasis on the backyard, Shorten to 150 words..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={chatLoading}
              />
              <button
                onClick={() => handleChatImprove(module.id)}
                disabled={chatLoading || !chatInput.trim()}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 font-semibold whitespace-nowrap"
              >
                {chatLoading ? 'Thinking...' : 'Send'}
              </button>
            </div>

            {/* AI Suggestion with Action Buttons */}
            {chatSuggestion && (
              <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-6">
                <div className="flex items-start justify-between mb-3">
                  <h5 className="font-semibold text-gray-900">AI Suggestion:</h5>
                  <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">New Version</span>
                </div>
                <div className="bg-white rounded-lg p-4 mb-4 max-h-64 overflow-y-auto">
                  <div className="whitespace-pre-wrap text-gray-900 leading-relaxed">
                    {chatSuggestion.replace(/\*\*/g, '')}
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => handleReplaceSuggestion(module.id)}
                    className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 font-semibold"
                  >
                    Replace Current
                  </button>
                  <button
                    onClick={() => {
                      setChatInput('');
                      handleChatImprove(module.id);
                    }}
                    className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 font-semibold"
                  >
                    Try Again
                  </button>
                  <button
                    onClick={handleRejectSuggestion}
                    className="bg-gray-200 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-300 font-semibold"
                  >
                    Reject
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderImages = () => {
    const designers = [
      {
        id: 'alessia_duval',
        name: 'Alessia Duval',
        style: 'Modern Minimalist',
        bio: 'Known for clean lines and neutral palettes, Alessia creates serene spaces that breathe elegance.',
        image: '/images/designers/alessia_duval.jpg'
      },
      {
        id: 'adrian_mercer',
        name: 'Adrian Mercer',
        style: 'Contemporary Luxe',
        bio: 'Bold textures and statement pieces define Adrian\'s sophisticated, modern aesthetic.',
        image: '/images/designers/adrian_mercer.jpg'
      },
      {
        id: 'lucien_hart',
        name: 'Lucien Hart',
        style: 'Classic Elegance',
        bio: 'Traditional craftsmanship meets timeless design in Lucien\'s refined interiors.',
        image: '/images/designers/lucien_hart.jpg'
      }
    ];

    const colorSchemes = [
      {
        id: 'glacial_muse',
        name: 'Glacial Muse',
        colors: ['#E8F4F8', '#B8D8E8', '#7BA8C0'],
        description: 'Cool, calming blues and whites create a serene, spa-like atmosphere'
      },
      {
        id: 'nomad_prism',
        name: 'Nomad Prism',
        colors: ['#D4A574', '#8B7355', '#E6D5C3'],
        description: 'Warm earth tones and desert-inspired hues for a bohemian feel'
      },
      {
        id: 'urban_alloy',
        name: 'Urban Alloy',
        colors: ['#4A4A4A', '#7D7D7D', '#A8A8A8'],
        description: 'Industrial grays and metallics for a modern, sophisticated look'
      },
      {
        id: 'sage_whisper',
        name: 'Sage Whisper',
        colors: ['#B8C5B0', '#8FA888', '#6B8E6B'],
        description: 'Soft greens and natural tones bring the outdoors in'
      },
      {
        id: 'terracotta_dream',
        name: 'Terracotta Dream',
        colors: ['#E07856', '#C65D3B', '#A0522D'],
        description: 'Warm, inviting oranges and browns for a cozy Mediterranean vibe'
      },
      {
        id: 'midnight_navy',
        name: 'Midnight Navy',
        colors: ['#1C3A57', '#2C5F8D', '#4A7BA7'],
        description: 'Deep, rich blues create drama and sophistication'
      }
    ];

    return (
      <div className="space-y-8">
        {/* Designer Gallery */}
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Choose Your Designer</h2>
          <div className="grid md:grid-cols-3 gap-6">
            {designers.map(designer => (
              <div key={designer.id} className="bg-white border-2 border-gray-200 rounded-lg overflow-hidden hover:border-blue-400 transition-all">
                <div className="h-64 bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-32 h-32 bg-white rounded-full mx-auto mb-4 flex items-center justify-center">
                      <span className="text-5xl">👤</span>
                    </div>
                    <h3 className="text-xl font-bold text-gray-900">{designer.name}</h3>
                  </div>
                </div>
                <div className="p-4">
                  <div className="text-sm font-semibold text-blue-600 mb-2">{designer.style}</div>
                  <p className="text-sm text-gray-600">{designer.bio}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Color Schemes */}
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Color Schemes</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {colorSchemes.map(scheme => (
              <div key={scheme.id} className="bg-white border-2 border-gray-200 rounded-lg p-4 hover:border-blue-400 transition-all">
                <div className="flex space-x-2 mb-3">
                  {scheme.colors.map((color, i) => (
                    <div
                      key={i}
                      className="flex-1 h-16 rounded"
                      style={{ backgroundColor: color }}
                    />
                  ))}
                </div>
                <h3 className="font-semibold text-gray-900 mb-1">{scheme.name}</h3>
                <p className="text-xs text-gray-600">{scheme.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Upload Section */}
        <div className="border-t border-gray-200 pt-8">
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Property Photos</h3>
            <div className="flex items-center space-x-4">
              <label className="cursor-pointer bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors">
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  disabled={uploadingImages}
                />
                {uploadingImages ? 'Uploading...' : '📤 Upload Images'}
              </label>
              <span className="text-sm text-gray-600">Select multiple images to upload at once</span>
            </div>
          </div>
        </div>

        {/* Images Grid */}
        {images.length > 0 ? (
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Property Images ({images.length})
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {images.map(image => (
                <div key={image.id} className="relative group">
                  <img
                    src={`${BACKEND_URL}${image.url}`}
                    alt={image.filename}
                    className="w-full h-48 object-cover rounded-lg"
                  />
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-opacity rounded-lg flex items-center justify-center">
                    <button
                      onClick={() => handleDeleteImage(image.id)}
                      className="opacity-0 group-hover:opacity-100 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-all"
                    >
                      Delete
                    </button>
                  </div>
                  {image.is_primary && (
                    <div className="absolute top-2 right-2 bg-blue-600 text-white text-xs px-2 py-1 rounded">
                      Primary
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-center py-12 bg-gray-50 rounded-lg">
            <span className="text-4xl mb-2 block">📸</span>
            <p className="text-gray-600">No images uploaded yet</p>
          </div>
        )}

        {/* AI Interior Design Section */}
        {images.length > 0 && (
          <div className="border-t border-gray-200 pt-6">
            {/* Info Banner */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <p className="text-sm text-blue-800">
                🎨 <strong>AI Interior Design:</strong> Select images, configure room type and design settings for each, then process with AI (5 credits per image)
              </p>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900">
                AI Interior Design ({selectedImages.length} selected)
              </h3>
              <button
                onClick={handleProcessInteriorDesign}
                disabled={selectedImages.length === 0 || processingDesign}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 font-semibold"
              >
                {processingDesign ? 'Processing...' : `Process ${selectedImages.length} Image${selectedImages.length !== 1 ? 's' : ''} (${selectedImages.length * 5} credits)`}
              </button>
            </div>

            {/* Image Grid with Per-Image Settings */}
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              {images.map(image => {
                const imageSettings = selectedImages.find(s => s.image_id === image.id);
                const isSelected = !!imageSettings;

                return (
                  <div
                    key={image.id}
                    className={`border-2 rounded-lg overflow-hidden transition-all ${
                      isSelected
                        ? 'border-blue-600 shadow-lg bg-blue-50'
                        : 'border-gray-200 bg-white hover:border-gray-300'
                    }`}
                  >
                    {/* Image */}
                    <div className="relative">
                      <img
                        src={`${BACKEND_URL}${image.url}`}
                        alt={image.filename}
                        className="w-full h-64 object-cover"
                      />
                      {isSelected && (
                        <div className="absolute top-2 right-2 bg-blue-600 text-white px-3 py-1 rounded-full font-semibold">
                          ✓ Selected
                        </div>
                      )}
                    </div>

                    {/* Settings */}
                    <div className="p-4 space-y-3">
                      {/* Room Type */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Room Type</label>
                        <select
                          value={imageSettings?.room_type || 'living_room'}
                          onChange={(e) => {
                            const newSettings = {
                              image_id: image.id,
                              room_type: e.target.value,
                              designer: imageSettings?.designer || 'alessia_duval',
                              color_scheme: imageSettings?.color_scheme || 'glacial_muse'
                            };
                            
                            if (isSelected) {
                              // Update existing
                              setSelectedImages(prev => 
                                prev.map(s => s.image_id === image.id ? newSettings : s)
                              );
                            } else {
                              // Add new
                              setSelectedImages(prev => [...prev, newSettings]);
                            }
                          }}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                        >
                          <option value="living_room">Living Room</option>
                          <option value="bedroom">Bedroom</option>
                          <option value="kitchen">Kitchen</option>
                          <option value="bathroom">Bathroom</option>
                          <option value="dining_room">Dining Room</option>
                          <option value="office">Office</option>
                          <option value="exterior">Exterior</option>
                        </select>
                      </div>

                      {/* Designer & Color Scheme (only show if selected) */}
                      {isSelected && (
                        <>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Designer Style</label>
                            <select
                              value={imageSettings.designer}
                              onChange={(e) => {
                                setSelectedImages(prev =>
                                  prev.map(s => s.image_id === image.id 
                                    ? {...s, designer: e.target.value}
                                    : s
                                  )
                                );
                              }}
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                            >
                              <option value="alessia_duval">Alessia Duval</option>
                              <option value="adrian_mercer">Adrian Mercer</option>
                              <option value="lucien_hart">Lucien Hart</option>
                            </select>
                          </div>

                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Color Scheme</label>
                            <select
                              value={imageSettings.color_scheme}
                              onChange={(e) => {
                                setSelectedImages(prev =>
                                  prev.map(s => s.image_id === image.id 
                                    ? {...s, color_scheme: e.target.value}
                                    : s
                                  )
                                );
                              }}
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                            >
                              <option value="glacial_muse">Glacial Muse</option>
                              <option value="nomad_prism">Nomad Prism</option>
                              <option value="urban_alloy">Urban Alloy</option>
                            </select>
                          </div>
                        </>
                      )}

                      {/* Select/Remove Button */}
                      <button
                        onClick={() => {
                          if (isSelected) {
                            setSelectedImages(prev => prev.filter(s => s.image_id !== image.id));
                          } else {
                            setSelectedImages(prev => [...prev, {
                              image_id: image.id,
                              room_type: 'living_room',
                              designer: 'alessia_duval',
                              color_scheme: 'glacial_muse'
                            }]);
                          }
                        }}
                        className={`w-full py-2 rounded-lg font-medium transition-colors ${
                          isSelected
                            ? 'bg-red-100 text-red-700 hover:bg-red-200'
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                        }`}
                      >
                        {isSelected ? 'Remove from Selection' : 'Select for Processing'}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Processed Designs */}
            {listing?.interior_design_variants?.length > 0 && (
              <div className="border-t border-gray-200 pt-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Processed Designs ({listing.interior_design_variants.length})
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {listing.interior_design_variants.map(variant => (
                    <div key={variant.id} className="border border-gray-200 rounded-lg overflow-hidden">
                      {variant.processed_image_url ? (
                        <img
                          src={`${BACKEND_URL}${variant.processed_image_url}`}
                          alt="Interior Design"
                          className="w-full h-48 object-cover"
                        />
                      ) : (
                        <div className="w-full h-48 bg-gray-100 flex items-center justify-center">
                          <div className="text-center">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                            <div className="text-sm text-gray-600">Processing...</div>
                            <div className="text-xs text-gray-500 mt-1">Check back in 2-3 minutes</div>
                          </div>
                        </div>
                      )}
                      <div className="p-3 bg-white">
                        <div className="text-xs text-gray-600">
                          {variant.room_type} • {variant.designer}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          {variant.status === 'completed' ? '✓ Completed' : '⏳ Processing...'}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Process Images Button at Bottom */}
            {selectedImages.length > 0 && (
              <div className="border-t border-gray-200 pt-6">
                <button
                  onClick={handleProcessInteriorDesign}
                  disabled={processingDesign}
                  className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all disabled:bg-gray-400 font-bold text-lg shadow-lg"
                >
                  {processingDesign ? 'Processing...' : `🎨 Process ${selectedImages.length} Image${selectedImages.length !== 1 ? 's' : ''} with AI Interior Design (${selectedImages.length * 5} Credits)`}
                </button>
                <p className="text-sm text-gray-600 text-center mt-2">
                  Processing takes 2-3 minutes per image. You'll be notified when complete.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  // renderInteriorDesign function removed - functionality integrated into renderImages

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

  const renderOverview = () => {
    if (!listing) return null;
    
    const foundationComplete = listing.foundation_status === 'completed';
    const foundationProcessing = listing.foundation_status === 'processing';
    const foundationModules = modules.filter(m => m.isFoundation);
    const dependentModules = modules.filter(m => m.requiresFoundation);
    const otherModules = modules.filter(m => !m.isFoundation && !m.requiresFoundation && !m.ai);

    return (
      <div className="space-y-8">
        {/* Property Details Card */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                {listing.property_details.address}
              </h2>
              <p className="text-gray-600 mb-4">
                {listing.property_details.city}, {listing.property_details.state} • 
                {listing.property_details.beds} bed, {listing.property_details.baths} bath • 
                {listing.property_details.sqft && ` ${listing.property_details.sqft.toLocaleString()} sq ft`}
              </p>
              <div className="flex items-center space-x-4">
                <div className="text-sm text-gray-600">
                  📸 {images.length} photos
                </div>
                <div className="text-sm text-gray-600">
                  ✅ {Object.keys(moduleContent).length} content pieces
                </div>
              </div>
            </div>
            {listing.property_details.listing_price && (
              <div className="text-right">
                <div className="text-sm text-gray-600">Listing Price</div>
                <div className="text-3xl font-bold text-blue-600">
                  ${listing.property_details.listing_price.toLocaleString()}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Foundation Status Banner */}
        {foundationProcessing && (
          <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-6">
            <div className="flex items-center space-x-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900">Generating Foundation Content...</h3>
                <p className="text-sm text-gray-600 mt-1">
                  We're researching your neighborhood, writing your property description, and analyzing the market. This takes about 1-2 minutes.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Foundation Tools */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-900">Foundation (Auto-Generated with 20 Credits)</h3>
            {foundationComplete && (
              <span className="bg-green-100 text-green-800 text-sm px-3 py-1 rounded-full font-medium">
                ✓ Complete
              </span>
            )}
          </div>
          <div className="grid md:grid-cols-3 gap-4">
            {foundationModules.map(module => {
              const isCompleted = moduleContent[module.id];
              return (
                <button
                  key={module.id}
                  onClick={() => {
                    if (isCompleted) {
                      setActiveModule(module.id);
                      setActiveView('module');
                    }
                  }}
                  disabled={!isCompleted}
                  className={`text-left p-6 rounded-lg border-2 transition-all ${
                    isCompleted
                      ? 'bg-green-50 border-green-300 hover:shadow-lg cursor-pointer'
                      : 'bg-gray-50 border-gray-200 cursor-not-allowed opacity-60'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <span className="text-4xl">{module.icon}</span>
                    {isCompleted ? (
                      <span className="bg-green-600 text-white text-xs px-2 py-1 rounded-full">
                        ✓ Done
                      </span>
                    ) : (
                      <span className="text-gray-400 text-xs">Generating...</span>
                    )}
                  </div>
                  <h4 className="font-semibold text-gray-900 mb-2">{module.name}</h4>
                  <p className="text-sm text-gray-600">{module.description}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Dependent Tools */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-900">Marketing & Content Tools</h3>
            {!foundationComplete && (
              <span className="bg-yellow-100 text-yellow-800 text-xs px-3 py-1 rounded-full font-medium">
                🔒 Unlocks after foundation
              </span>
            )}
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {dependentModules.map(module => {
              const isCompleted = moduleContent[module.id];
              const isLocked = !foundationComplete;
              
              return (
                <button
                  key={module.id}
                  onClick={() => {
                    if (!isLocked) {
                      setActiveModule(module.id);
                      setActiveView('module');
                    }
                  }}
                  disabled={isLocked}
                  className={`text-left p-6 rounded-lg border-2 transition-all ${
                    isLocked
                      ? 'bg-gray-50 border-gray-200 cursor-not-allowed opacity-50'
                      : isCompleted
                      ? 'bg-green-50 border-green-300 hover:shadow-lg'
                      : 'bg-white border-gray-200 hover:border-blue-300 hover:shadow-lg'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <span className="text-4xl">{module.icon}</span>
                    {isLocked ? (
                      <span className="text-gray-400 text-2xl">🔒</span>
                    ) : isCompleted ? (
                      <span className="bg-green-600 text-white text-xs px-2 py-1 rounded-full">
                        ✓ Done
                      </span>
                    ) : null}
                  </div>
                  <h4 className="font-semibold text-gray-900 mb-2">{module.name}</h4>
                  <p className="text-sm text-gray-600 mb-3">{module.description}</p>
                  {!isLocked && (
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-blue-600">
                        {module.credits} credit
                      </span>
                      <span className="text-xs text-gray-500">
                        {isCompleted ? 'View / Edit' : 'Generate →'}
                      </span>
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Image & Design Tools */}
        <div>
          <h3 className="text-xl font-bold text-gray-900 mb-4">Images & Design</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {otherModules.map(module => (
              <button
                key={module.id}
                onClick={() => {
                  setActiveModule(module.id);
                  setActiveView('module');
                }}
                className="text-left p-6 rounded-lg border-2 bg-white border-gray-200 hover:border-blue-300 transition-all hover:shadow-lg"
              >
                <div className="flex items-start justify-between mb-3">
                  <span className="text-4xl">{module.icon}</span>
                  {module.id === 'images' && images.length > 0 && (
                    <span className="bg-blue-600 text-white text-xs px-2 py-1 rounded-full">
                      {images.length}
                    </span>
                  )}
                </div>
                <h4 className="font-semibold text-gray-900 mb-2">{module.name}</h4>
                <p className="text-sm text-gray-600 mb-3">{module.description}</p>
                {module.credits && (
                  <span className="text-xs font-medium text-blue-600">
                    {module.credits} credits per image
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderDetails_Old = () => {
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
  const foundationComplete = listing.foundation_status === 'completed';
  const foundationProcessing = listing.foundation_status === 'processing';
  
  // Group modules by category
  const modulesByCategory = modules.reduce((acc, module) => {
    if (!acc[module.category]) acc[module.category] = [];
    acc[module.category].push(module);
    return acc;
  }, {});

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="bg-white shadow z-10">
        <div className="max-w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={onBack}
                className="text-gray-600 hover:text-gray-900"
              >
                ← Back to Listings
              </button>
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  {listing.property_details.address}
                </h1>
                <p className="text-sm text-gray-600">
                  {listing.property_details.city}, {listing.property_details.state}
                </p>
              </div>
            </div>
            <div className="text-sm">
              <div className="text-gray-600">Credits</div>
              <div className="text-2xl font-bold text-blue-600">{user?.credits || 0}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Sidebar + Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Tools */}
        <div className="w-80 bg-white border-r border-gray-200 overflow-y-auto">
          <div className="p-4">
            {/* Foundation Status */}
            {foundationProcessing && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <span className="text-sm text-blue-800 font-medium">Generating foundation...</span>
                </div>
              </div>
            )}
            
            {/* Tool Categories */}
            {Object.entries(modulesByCategory).map(([category, categoryModules]) => (
              <div key={category} className="mb-6">
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 px-2">
                  {category}
                </h3>
                <div className="space-y-1">
                  {categoryModules.map(module => {
                    const isCompleted = moduleContent[module.id];
                    const isLocked = module.requiresFoundation && !foundationComplete;
                    const isActive = activeModule === module.id;
                    
                    return (
                      <button
                        key={module.id}
                        onClick={() => {
                          if (!isLocked) {
                            setActiveModule(module.id);
                            setActiveView('module');
                          }
                        }}
                        disabled={isLocked}
                        className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                          isActive
                            ? 'bg-blue-50 border-2 border-blue-500 text-blue-700'
                            : isLocked
                            ? 'bg-gray-50 text-gray-400 cursor-not-allowed'
                            : isCompleted
                            ? 'hover:bg-green-50 text-gray-900'
                            : 'hover:bg-gray-100 text-gray-700'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2 flex-1 min-w-0">
                            <span className="text-lg flex-shrink-0">{module.icon}</span>
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium truncate">{module.name}</div>
                              {module.credits > 0 && !isLocked && (
                                <div className="text-xs text-gray-500">{module.credits} credit</div>
                              )}
                            </div>
                          </div>
                          <div className="flex-shrink-0 ml-2">
                            {isLocked ? (
                              <span className="text-gray-400">🔒</span>
                            ) : isCompleted ? (
                              <span className="text-green-600">✓</span>
                            ) : null}
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Content Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-8">
            {!activeModule ? (
              <div className="text-center py-20">
                <div className="w-32 h-32 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <span className="text-6xl">🏠</span>
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  {listing.property_details.address}
                </h2>
                <p className="text-lg text-gray-600 mb-2">
                  {listing.property_details.beds} bed • {listing.property_details.baths} bath • 
                  {listing.property_details.sqft && ` ${listing.property_details.sqft.toLocaleString()} sq ft`}
                </p>
                {listing.property_details.listing_price && (
                  <p className="text-3xl font-bold text-blue-600 mb-8">
                    ${listing.property_details.listing_price.toLocaleString()}
                  </p>
                )}
                <p className="text-gray-600 mb-4">
                  Select a tool from the left sidebar to get started
                </p>
                {foundationProcessing && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 inline-block">
                    <p className="text-sm text-blue-800">
                      Foundation content is being generated. This takes 1-2 minutes.
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center space-x-3">
                    <span className="text-4xl">{currentModule?.icon}</span>
                    <div>
                      <h2 className="text-2xl font-bold text-gray-900">{currentModule?.name}</h2>
                      <p className="text-sm text-gray-600">{currentModule?.description}</p>
                    </div>
                  </div>
                  {currentModule?.credits > 0 && (
                    <div className="text-right">
                      <div className="text-xs text-gray-500">Cost</div>
                      <div className="text-lg font-bold text-blue-600">{currentModule.credits} credit{currentModule.credits > 1 ? 's' : ''}</div>
                    </div>
                  )}
                </div>
                
                {activeModule === 'images' ? renderImages() : renderModuleContent(currentModule)}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default IndividualListingPage;
