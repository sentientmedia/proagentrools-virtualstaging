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
    { id: 'listing_copy', name: 'Listing Description', icon: '✍️', ai: true, credits: 1, description: 'Generate professional property listing description' },
    { id: 'marketing_copy', name: 'Marketing Copy', icon: '📢', ai: true, credits: 1, description: 'Create compelling marketing materials' },
    { id: 'social_media', name: 'Social Media Posts', icon: '📱', ai: true, credits: 1, description: 'Generate engaging social media content' },
    { id: 'email_template', name: 'Email Templates', icon: '✉️', ai: true, credits: 1, description: 'Create professional email templates' },
    { id: 'market_intel', name: 'Market Intelligence', icon: '📊', ai: true, credits: 1, description: 'Get market analysis and positioning' },
    { id: 'virtual_tour_script', name: 'Virtual Tour Script', icon: '🎥', ai: true, credits: 1, description: 'Create walkthrough video script' },
    { id: 'images', name: 'Property Images', icon: '📸', description: 'Upload and manage property photos' },
    { id: 'interior_design', name: 'AI Interior Design', icon: '🎨', credits: 5, description: 'Transform photos with AI staging' },
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

  const renderImages = () => {
    return (
      <div className="space-y-6">
        {/* Upload Section */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Images</h3>
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
      </div>
    );
  };

  const renderInteriorDesign = () => {
    return (
      <div className="space-y-6">
        {/* Selection Info */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <p className="text-sm text-blue-800">
            💡 Select images from your property photos, choose design settings, and process them with AI interior design (5 credits per image)
          </p>
        </div>

        {images.length === 0 ? (
          <div className="text-center py-12 bg-gray-50 rounded-lg">
            <span className="text-4xl mb-2 block">🎨</span>
            <p className="text-gray-600 mb-4">Upload images first to use interior design</p>
            <button
              onClick={() => setActiveModule('images')}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
            >
              Go to Images
            </button>
          </div>
        ) : (
          <>
            {/* Design Settings */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Design Settings</h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Room Type</label>
                  <select
                    value={designSettings.room_type}
                    onChange={(e) => setDesignSettings({...designSettings, room_type: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                  >
                    <option value="living_room">Living Room</option>
                    <option value="bedroom">Bedroom</option>
                    <option value="kitchen">Kitchen</option>
                    <option value="bathroom">Bathroom</option>
                    <option value="dining_room">Dining Room</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Designer Style</label>
                  <select
                    value={designSettings.designer}
                    onChange={(e) => setDesignSettings({...designSettings, designer: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                  >
                    <option value="alessia_duval">Alessia Duval</option>
                    <option value="adrian_mercer">Adrian Mercer</option>
                    <option value="lucien_hart">Lucien Hart</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Color Scheme</label>
                  <select
                    value={designSettings.color_scheme}
                    onChange={(e) => setDesignSettings({...designSettings, color_scheme: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                  >
                    <option value="glacial_muse">Glacial Muse</option>
                    <option value="nomad_prism">Nomad Prism</option>
                    <option value="urban_alloy">Urban Alloy</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Image Selection */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  Select Images to Process ({selectedImageIds.length} selected)
                </h3>
                <button
                  onClick={handleProcessInteriorDesign}
                  disabled={selectedImageIds.length === 0 || processingDesign}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400"
                >
                  {processingDesign ? 'Processing...' : `Process Selected (${selectedImageIds.length * 5} credits)`}
                </button>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {images.map(image => (
                  <div
                    key={image.id}
                    onClick={() => {
                      setSelectedImageIds(prev =>
                        prev.includes(image.id)
                          ? prev.filter(id => id !== image.id)
                          : [...prev, image.id]
                      );
                    }}
                    className={`relative cursor-pointer border-4 rounded-lg transition-all ${
                      selectedImageIds.includes(image.id)
                        ? 'border-blue-600 shadow-lg'
                        : 'border-transparent hover:border-gray-300'
                    }`}
                  >
                    <img
                      src={`${BACKEND_URL}${image.url}`}
                      alt={image.filename}
                      className="w-full h-48 object-cover rounded"
                    />
                    {selectedImageIds.includes(image.id) && (
                      <div className="absolute top-2 right-2 bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center">
                        ✓
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Interior Design Variants */}
            {listing?.interior_design_variants?.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Processed Designs ({listing.interior_design_variants.length})
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {listing.interior_design_variants.map(variant => (
                    <div key={variant.id} className="border border-gray-200 rounded-lg overflow-hidden">
                      <img
                        src={`${BACKEND_URL}${variant.processed_image_url}`}
                        alt="Interior Design"
                        className="w-full h-48 object-cover"
                      />
                      <div className="p-3 bg-white">
                        <div className="text-xs text-gray-600">
                          {variant.designer} • {variant.color_scheme}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
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

  const renderOverview = () => {
    if (!listing) return null;

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
                  ✅ {Object.keys(moduleContent).length} tools completed
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

        {/* AI Tools Grid */}
        <div>
          <h3 className="text-xl font-bold text-gray-900 mb-4">AI-Powered Tools</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {modules.filter(m => m.ai).map(module => {
              const isCompleted = moduleContent[module.id];
              return (
                <button
                  key={module.id}
                  onClick={() => {
                    setActiveModule(module.id);
                    setActiveView('module');
                  }}
                  className={`text-left p-6 rounded-lg border-2 transition-all hover:shadow-lg ${
                    isCompleted
                      ? 'bg-green-50 border-green-300'
                      : 'bg-white border-gray-200 hover:border-blue-300'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <span className="text-4xl">{module.icon}</span>
                    {isCompleted && (
                      <span className="bg-green-600 text-white text-xs px-2 py-1 rounded-full">
                        ✓ Done
                      </span>
                    )}
                  </div>
                  <h4 className="font-semibold text-gray-900 mb-2">{module.name}</h4>
                  <p className="text-sm text-gray-600 mb-3">{module.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-blue-600">
                      {module.credits} credit{module.credits > 1 ? 's' : ''}
                    </span>
                    <span className="text-xs text-gray-500">
                      {isCompleted ? 'View / Edit' : 'Generate →'}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Image & Design Tools */}
        <div>
          <h3 className="text-xl font-bold text-gray-900 mb-4">Images & Design</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {modules.filter(m => !m.ai).map(module => (
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

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => {
                  if (activeView === 'module') {
                    setActiveView('overview');
                    setActiveModule(null);
                  } else {
                    onBack();
                  }
                }}
                className="text-gray-600 hover:text-gray-900"
              >
                ← {activeView === 'module' ? 'Back to Tools' : 'Back to Listings'}
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

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeView === 'overview' && renderOverview()}
        
        {activeView === 'module' && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              {currentModule?.icon} {currentModule?.name}
            </h2>
            
            {activeModule === 'images' && renderImages()}
            {activeModule === 'interior_design' && renderInteriorDesign()}
            {currentModule?.ai && renderModuleContent(currentModule)}
          </div>
        )}
      </div>
    </div>
  );
};

export default IndividualListingPage;
