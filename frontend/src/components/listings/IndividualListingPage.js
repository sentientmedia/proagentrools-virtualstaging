import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import axios from 'axios';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import ModuleQuestionnaireModal from './ModuleQuestionnaireModal';

// Fix Leaflet default icon issue - Use CDN URLs to bypass webpack issues
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

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
  const [selectedImages, setSelectedImages] = useState([]); // Array of {image_id, room_type}
  
  // Global interior design settings (applied to all selected images)
  const [globalDesigner, setGlobalDesigner] = useState('alessia_duval');
  const [globalColorScheme, setGlobalColorScheme] = useState('glacial_muse');
  const [customDescription, setCustomDescription] = useState('');
  const [customColors, setCustomColors] = useState('');
  const [processingDesign, setProcessingDesign] = useState(false);
  
  // Designer bio modal
  const [selectedDesigner, setSelectedDesigner] = useState(null);
  const [showDesignerModal, setShowDesignerModal] = useState(false);
  
  // Foundation lock modal
  const [showFoundationModal, setShowFoundationModal] = useState(false);
  
  // Map coordinates state
  const [mapCoordinates, setMapCoordinates] = useState(null);
  const [geocodingAddress, setGeocodingAddress] = useState(false);
  
  // Trails state
  const [trails, setTrails] = useState([]);
  const [loadingTrails, setLoadingTrails] = useState(false);
  
  // Module questionnaire modal state
  const [showQuestionnaireModal, setShowQuestionnaireModal] = useState(false);
  const [questionnaireModule, setQuestionnaireModule] = useState(null);
  const [writingStyles, setWritingStyles] = useState([]);
  const [formData, setFormData] = useState({
    tone: '',
    target_buyer_type: '',
    property_highlights: [],
    competitive_advantages: [],
    open_house_date: '',
    open_house_time: '',
    open_house_features: [],
    video_length: '',
    rooms_to_highlight: [],
    pricing_strategy: '',
    recent_upgrades: [],
    known_objections: [],
    showing_feedback: [],
    additional_context: ''
  });

  const modules = [
    // Foundation modules (auto-generated, always unlocked)
    { id: 'neighborhood_research', name: 'Neighborhood Research', icon: '🏘️', ai: true, credits: 0, description: 'Detailed neighborhood analysis', isFoundation: true, category: 'Foundation' },
    { id: 'listing_copy', name: 'Property Description', icon: '✍️', ai: true, credits: 0, description: 'Professional listing copy', isFoundation: true, category: 'Foundation' },
    { id: 'market_intel', name: 'Market Intelligence', icon: '📊', ai: true, credits: 0, description: 'Strategic market positioning', isFoundation: true, category: 'Foundation' },
    
    // Media - AI Photo Editor (moved to top)
    { id: 'images', name: 'Photos & Interior Design', icon: '📸', description: 'Upload and enhance images', category: 'Media' },
    
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
  ];

  useEffect(() => {
    if (listingId) {
      loadListing();
      loadImages();
      loadWritingStyles();
    }
  }, [listingId]);
  
  const loadWritingStyles = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/profile/writing-styles`);
      setWritingStyles(response.data.styles || []);
    } catch (err) {
      console.error('Failed to load writing styles:', err);
    }
  };
  
  const loadTrails = async () => {
    try {
      setLoadingTrails(true);
      const response = await axios.get(`${BACKEND_URL}/api/listings/${listingId}/trails`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setTrails(response.data.trails || []);
    } catch (err) {
      console.error('Failed to load trails:', err);
      setTrails([]);
    } finally {
      setLoadingTrails(false);
    }
  };
  
  // Geocode address to get coordinates
  const geocodeAddress = async (property_details) => {
    if (geocodingAddress) return; // Prevent duplicate requests
    
    // If we already have coordinates, use them
    if (property_details.latitude && property_details.longitude) {
      setMapCoordinates({
        lat: property_details.latitude,
        lng: property_details.longitude
      });
      return;
    }
    
    setGeocodingAddress(true);
    
    try {
      // Build full address string
      const fullAddress = `${property_details.address}, ${property_details.city}, ${property_details.state} ${property_details.zip_code}`;
      
      // Use Nominatim (OpenStreetMap) geocoding service
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(fullAddress)}&limit=1`,
        {
          headers: {
            'User-Agent': 'ProAgentTools/1.0' // Required by Nominatim
          }
        }
      );
      
      const data = await response.json();
      
      if (data && data.length > 0) {
        const { lat, lon } = data[0];
        setMapCoordinates({
          lat: parseFloat(lat),
          lng: parseFloat(lon)
        });
        
        console.log(`✅ Geocoded address: ${fullAddress} -> [${lat}, ${lon}]`);
      } else {
        // Fallback to city/state if full address doesn't work
        const cityStateAddress = `${property_details.city}, ${property_details.state}`;
        const fallbackResponse = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(cityStateAddress)}&limit=1`,
          {
            headers: {
              'User-Agent': 'ProAgentTools/1.0'
            }
          }
        );
        
        const fallbackData = await fallbackResponse.json();
        
        if (fallbackData && fallbackData.length > 0) {
          const { lat, lon } = fallbackData[0];
          setMapCoordinates({
            lat: parseFloat(lat),
            lng: parseFloat(lon)
          });
          
          console.log(`⚠️ Used city geocoding: ${cityStateAddress} -> [${lat}, ${lon}]`);
        } else {
          // Last resort: default coordinates
          console.warn('❌ Geocoding failed, using default coordinates');
          setMapCoordinates({ lat: 39.8283, lng: -98.5795 });
        }
      }
    } catch (error) {
      console.error('Geocoding error:', error);
      // Use default coordinates on error
      setMapCoordinates({ lat: 39.8283, lng: -98.5795 });
    } finally {
      setGeocodingAddress(false);
    }
  };

  useEffect(() => {
    if (listingId) {
      loadListing();
      loadImages();
    }
  }, [listingId]);

  // Auto-refresh foundation status if processing
  useEffect(() => {
    if (!listing || listing.foundation_status !== 'processing') return;
    
    const checkFoundation = setInterval(() => {
      loadListing();
    }, 5000); // Check every 5 seconds
    
    return () => clearInterval(checkFoundation);
  }, [listing?.foundation_status]);
  
  // Geocode address when listing loads
  useEffect(() => {
    if (listing && listing.property_details && !mapCoordinates && !geocodingAddress) {
      geocodeAddress(listing.property_details);
    }
  }, [listing]);

  // Auto-refresh for processing interior designs
  useEffect(() => {
    if (!listing || !listing.interior_design_variants) return;
    
    // Check if any variants are processing
    const hasProcessing = listing.interior_design_variants.some(v => v.status === 'processing');
    
    if (!hasProcessing) return;
    
    const checkDesigns = setInterval(() => {
      loadListing();
    }, 5000); // Check every 5 seconds
    
    return () => clearInterval(checkDesigns);
  }, [listing?.interior_design_variants]);

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

  const openQuestionnaire = (moduleName) => {
    // Reset form data
    setFormData({
      tone: '',
      target_buyer_type: '',
      property_highlights: [],
      competitive_advantages: [],
      open_house_date: '',
      open_house_time: '',
      open_house_features: [],
      video_length: '',
      rooms_to_highlight: [],
      pricing_strategy: '',
      recent_upgrades: [],
      known_objections: [],
      showing_feedback: [],
      additional_context: ''
    });
    
    setQuestionnaireModule(moduleName);
    setShowQuestionnaireModal(true);
  };
  
  const handleGenerateContent = async (moduleName, formInputs = null) => {
    try {
      setGeneratingModule(moduleName);
      
      // Determine which modules to include as context
      const includeModuleContext = [];
      
      // Always include foundation modules if they exist
      const foundationModules = ['neighborhood_research', 'listing_copy', 'market_intel'];
      for (const foundationId of foundationModules) {
        if (moduleContent[foundationId] && moduleContent[foundationId].content) {
          includeModuleContext.push(foundationId);
        }
      }
      
      // Include other completed modules that come "before" this one in the logical flow
      // This creates a cascade effect where later modules build on earlier ones
      const moduleOrder = [
        'neighborhood_research', 'listing_copy', 'market_intel', // Foundation
        'buyer_profile', 'price_justification', 'competitor_comparison', // Analysis first
        'marketing_copy', 'social_media', 'email_template', // Then marketing
        'property_highlights', 'agent_talking_points', 'objection_handling', // Sales support
        'virtual_tour_script', 'open_house_promo', 'negotiation_tips', // Advanced
        'seller_updates', 'buyer_followup' // Communications
      ];
      
      const currentModuleIndex = moduleOrder.indexOf(moduleName);
      if (currentModuleIndex > 0) {
        // Include all previously completed modules in order
        for (let i = 0; i < currentModuleIndex; i++) {
          const priorModuleId = moduleOrder[i];
          if (moduleContent[priorModuleId] && moduleContent[priorModuleId].content) {
            if (!includeModuleContext.includes(priorModuleId)) {
              includeModuleContext.push(priorModuleId);
            }
          }
        }
      }
      
      // Prepare request payload
      const payload = { 
        module_name: moduleName,
        include_module_context: includeModuleContext.length > 0 ? includeModuleContext : null,
        ...formInputs // Spread form inputs if provided
      };
      
      const response = await axios.post(
        `${BACKEND_URL}/api/listings/${listingId}/modules/${moduleName}/generate`,
        payload,
        { headers: { 'Authorization': `Bearer ${token}` } }
      );

      if (response.data.success) {
        setModuleContent(prev => ({
          ...prev,
          [moduleName]: {
            content: response.data.content,
            generated_at: new Date().toISOString(),
            is_ai_generated: true,
            used_context_from: includeModuleContext // Track which modules were used as context
          }
        }));
        
        // Reload listing to get updated module_outputs
        loadListing();
        
        // Close modal if open
        setShowQuestionnaireModal(false);
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

      // Apply global settings to all selected images
      const imagesWithSettings = selectedImages.map(img => ({
        image_id: img.image_id,
        room_type: img.room_type,
        designer: globalDesigner,
        color_scheme: globalColorScheme,
        custom_description: customDescription || undefined,
        custom_colors: customColors || undefined
      }));

      const response = await axios.post(
        `${BACKEND_URL}/api/listings/${listingId}/interior-design/process`,
        {
          images: imagesWithSettings
        },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );

      if (response.data.success) {
        alert(`${response.data.message}\n\nProcessed: ${response.data.processed_count} image(s)`);
        setSelectedImages([]);
        setCustomDescription('');
        setCustomColors('');
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

  const handleRerunDesign = async (variant) => {
    if (!window.confirm(`Rerun interior design for this image? This will cost 5 credits and regenerate using the same settings:\n\nRoom: ${variant.room_type}\nDesigner: ${variant.designer}\nColor: ${variant.color_scheme}`)) {
      return;
    }

    try {
      // Find the original image
      const originalImage = images.find(img => img.id === variant.original_image_id);
      if (!originalImage) {
        alert('Original image not found');
        return;
      }

      // Process with the same settings
      const response = await axios.post(
        `${BACKEND_URL}/api/listings/${listingId}/interior-design/process`,
        {
          images: [{
            image_id: variant.original_image_id,
            room_type: variant.room_type,
            designer: variant.designer || globalDesigner,
            color_scheme: variant.color_scheme || globalColorScheme,
            custom_description: customDescription || undefined,
            custom_colors: customColors || undefined
          }]
        },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );

      if (response.data.success) {
        alert('Rerunning interior design! Check back in 2-3 minutes.');
        await loadListing();
        await loadImages();
      }
    } catch (err) {
      console.error('Failed to rerun interior design:', err);
      alert(err.response?.data?.detail || 'Failed to rerun design');
    }
  };

  const renderModuleContent = (module) => {
    const content = moduleContent[module.id];

    if (!content) {
      if (module.ai) {
        // Determine which modules will be used as context for this generation
        const foundationModules = ['neighborhood_research', 'listing_copy', 'market_intel'];
        const contextModules = foundationModules.filter(fId => 
          moduleContent[fId] && moduleContent[fId].content
        );
        
        return (
          <div className="text-center py-12">
            <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-4xl">{module.icon}</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Generate {module.name}</h3>
            <p className="text-gray-600 mb-4">Use AI to generate professional content for this module</p>
            
            {contextModules.length > 0 && (
              <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg max-w-md mx-auto">
                <p className="text-sm font-semibold text-blue-900 mb-2">
                  🔗 This will build upon:
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {contextModules.map(moduleId => {
                    const moduleNames = {
                      'neighborhood_research': 'Neighborhood Research',
                      'listing_copy': 'Property Description',
                      'market_intel': 'Market Intelligence'
                    };
                    return (
                      <span key={moduleId} className="inline-block px-3 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                        {moduleNames[moduleId]}
                      </span>
                    );
                  })}
                </div>
                <p className="text-xs text-blue-700 mt-2">
                  Content will be consistent with previously generated modules
                </p>
              </div>
            )}
            
            <button
              onClick={() => openQuestionnaire(module.id)}
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
            <div className="bg-white border border-gray-200 rounded-lg p-8">
              <div className="prose prose-lg max-w-none">
                <div className="text-gray-800 leading-relaxed space-y-4" style={{ whiteSpace: 'pre-wrap' }}>
                  {content.content.split('\n\n').map((paragraph, idx) => {
                    // Handle headers (lines starting with ##)
                    if (paragraph.trim().startsWith('## ')) {
                      return (
                        <h2 key={idx} className="text-2xl font-bold text-gray-900 mt-6 mb-3">
                          {paragraph.replace(/^##\s*/, '')}
                        </h2>
                      );
                    }
                    // Handle headers (lines starting with #)
                    if (paragraph.trim().startsWith('# ')) {
                      return (
                        <h1 key={idx} className="text-3xl font-bold text-gray-900 mt-8 mb-4">
                          {paragraph.replace(/^#\s*/, '')}
                        </h1>
                      );
                    }
                    // Handle bullet points
                    if (paragraph.trim().startsWith('- ') || paragraph.trim().startsWith('* ')) {
                      const items = paragraph.split('\n').filter(line => line.trim());
                      return (
                        <ul key={idx} className="list-disc list-inside space-y-2 ml-4">
                          {items.map((item, i) => (
                            <li key={i} className="text-gray-800">
                              {item.replace(/^[-*]\s*/, '').replace(/\*\*/g, '')}
                            </li>
                          ))}
                        </ul>
                      );
                    }
                    // Regular paragraphs
                    return (
                      <p key={idx} className="text-gray-800 leading-relaxed">
                        {paragraph.replace(/\*\*/g, '')}
                      </p>
                    );
                  })}
                </div>
              </div>
            </div>
            <div className="flex items-center justify-between mt-2">
              <div className="flex items-center space-x-3">
                <div className="text-xs text-gray-500">
                  {content.is_ai_generated ? '🤖 AI Generated' : '✏️ Manually Edited'} • 
                  Last updated: {new Date(content.last_edited || content.generated_at).toLocaleString()}
                </div>
                {content.used_context_from && content.used_context_from.length > 0 && (
                  <div className="flex items-center space-x-1">
                    <span className="text-xs text-blue-600 font-medium">🔗</span>
                    <span className="text-xs text-blue-600">
                      Built upon {content.used_context_from.length} module{content.used_context_from.length > 1 ? 's' : ''}
                    </span>
                  </div>
                )}
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
                  onClick={() => openQuestionnaire(module.id)}
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
      { id: 'alessia_duval', name: 'Alessia Duval', style: 'Parisian Eclecticism', bio: 'Infuses Parisian elegance with global eclecticism, layering vibrant textiles and refined antiques.', image: '/images/designers/alessia_duval.jpg', full_bio: 'Born in the culturally diverse Marais district of Paris to a French antiques dealer mother and a father who was a Brazilian diplomat, Alessia Duval\'s earliest memories are colored with the international vibrancy of her upbringing. As a child, Alessia rarely lived in one place long; the diplomatic career of her father meant occasional relocations between Europe, South America, and North Africa.\n\nFrom each country, her mother encouraged her to collect handmade objects, exquisite textiles, and traditional crafts—small treasures carefully packed into aged trunks as they traveled onward. The foundational environment of her childhood cultivated Alessia\'s sophisticated yet eclectic palate.\n\nAlessia pursued formal education at La Cambre National School of Visual Arts in Brussels, where she developed her rich visual vocabulary, blending historical elegance with contemporary lines. After graduation, she worked briefly in prestigious design studios in Antwerp and Milan.\n\nHer style evolved uniquely shaped by cross-cultural exchange. The intense textile colors of India left a lasting imprint. Japanese minimal simplicity taught her restraint. North Africa influenced the visual flow and geometry of her spaces. Her South American experiences inspired tactile warmth and rich storytelling through natural materials.\n\nAlessia describes her design philosophy as "visual anthropology," aiming to narrate global journeys and cultural dialogues through carefully orchestrated spatial storytelling—each interior is an experience of travel distilled into a harmonious environment.' },
      { id: 'adrian_mercer', name: 'Adrian Mercer', style: 'Industrial Poetry', bio: 'Transforms post-industrial materials into poetic, sculptural interiors full of moody drama.', image: '/images/designers/adrian_mercer.jpg', full_bio: 'Born in Glasgow in 1985, Mercer grew up amid the city\'s post-industrial shipyards and steel factories. His mother, a sculptor, and his father, who ran a metal-recycling yard, filled his childhood with oxidized copper plates, rusted gears, and battered steel fragments that seeded his future design language.\n\nAt Central Saint Martins, Mercer immersed himself in material experimentation and digital fabrication, fearlessly mixing traditional craft techniques with modern technology. Influenced by welded and molded industrial materials transformed into objects of refinement, he delved deeper into texture-driven sculptural aesthetics.\n\nAfter graduation, travels across Europe and Asia broadened his design vocabulary. Early projects—sculptural lamps, tables, and chairs—quickly raised his profile on returning to London.\n\nIn 2014, Mercer founded "Mercer & Co." in a converted docklands warehouse, echoing his Glasgow roots. The studio became a playground for artisans, technologists, and sculptors, united by Mercer\'s credo: each project must tell a story, letting its industrial origins shine through even the most polished finishes.\n\nMercer draws on architects Tadao Ando and Zaha Hadid for material interplay and fluid dynamism. Brutalism and Cubism for angular forms. Global metalwork for tactile richness. Scotland\'s craggy coastlines inspire his palettes of oxidized copper, tarnished silver, deep greys, and sea-blues.' },
      { id: 'lucien_hart', name: 'Lucien Hart', style: 'Couture Glamour', bio: 'Fuses runway glamour with architectural audacity, delivering bold, theatrical spaces saturated in jewel tones.', image: '/images/designers/lucien_hart.jpg', full_bio: 'Born in Bordeaux (1985) to a fashion-stylist mother and architect father, Hart grew up scavenging ateliers and construction sites for inspiration. Summers in Paris revealed couture\'s backstage artistry; evenings at home taught him to sketch steel beams with the same reverence as silk drapery.\n\nAt Politecnico di Milano and later the Royal College of Art, Hart blurred clothing and architecture—installations that wrapped spaces in fabric, rooms that wore shadows like veils. London\'s avant-garde circles took note; his graduation pieces—sculptural lamps and leather-clad lounges—sold before the paint dried.\n\nFashion Inspirations: Alexander McQueen\'s narrative tension, Thierry Mugler\'s armored silhouettes, Tom Ford\'s sensual precision.\nFilm: Kubrick\'s unsettling rigor, Ridley Scott\'s dystopian density, Baz Luhrmann\'s decadent swoon.\nMovements: Vienna Secession unity, Art Deco exuberance.\nMaterials: Black marble, gilded steel, opalescent leather, digital light.\n\nHart launched Hart Atelier (London, with satellites in Paris and NYC) to craft "spaces that breathe drama." Notable commissions include Maison Saint Laurent Flagship in Paris, "Azure Twilight" Residence in Corsica, and Nocturnal Hotel in New York.\n\nHis credo: space is performance—luxury as rebellion, narrative as structure, emotion as material.' },
      { id: 'elinor_hartwell', name: 'Elinor Hartwell', style: 'Mindful Comfort', bio: 'Creates warm, soulful rooms where mindful living meets tactile, handcrafted comfort.', image: '/images/designers/elinor_hartwell.jpg', full_bio: 'Elinor Hartwell grew up in a windswept Devon village, where her family\'s seaside B&B and her mother\'s handmade pottery taught her that beauty lives in texture and imperfection. Mornings wandering salt-air gardens with her father, afternoons pressing clay—these rhythms shaped her intuitive grasp of material and mood.\n\nAfter studying interior architecture at the Royal College of Art, Hartwell honed her craft in Scandinavian studios, absorbing a refined simplicity that she melds with British warmth. Back in Brighton, she founded Hartwell Interiors, insisting every room feel "genuinely lived in": soft-edged alcoves for quiet reflection, communal nooks for laughter, surfaces that invite touch.\n\nHer hallmark is a tactile palette—reclaimed timber, hand-woven linen, raw clay ceramics—layered in muted creams, sage greens, storm-cloud grays. She commissions local artisans to create pieces that bear fingerprints and tiny flaws, turning honesty into luxury.\n\nIn Cornwall\'s Cove Hotel, Hartwell blurred indoors with wave-tumbled stone and driftwood accents. At Sussex\'s Rosewood Retreat, she converted a Victorian farmhouse into a wellness haven. Back home in Devon, The Hartwell Cottage—her childhood B&B reborn—melds vintage heirlooms with streamlined modernity.\n\nHartwell\'s designs feel less like décor and more like atmospheres—quiet sanctuaries that cultivate mindfulness, emotional ease, and the slow joy of simply being.' },
      { id: 'bianca_morelli', name: 'Bianca Morelli', style: 'Organic Elegance', bio: 'Weaves fluid, organic forms and tactile layers into emotionally engaging, elegant interiors.', image: '/images/designers/bianca_morelli.jpg', full_bio: 'Raised amid Florence\'s palaces and her mother\'s textile atelier, Morelli learned early that color and touch speak before words. Her father\'s architectural blueprints taught her form and function as a duet. Summers spent trailing through Renaissance gardens instilled a love for curves and layered light.\n\nAfter refining her vision at Politecnico di Milano, Morelli journeyed to Japan and Scandinavia, where she distilled minimalism into warmth and precision into ease. She absorbed the disciplined grace of Kyoto\'s temples and the soft functionality of Nordic fjord-side homes.\n\nIn her hands, materials awaken. Venetian glass sighs in undulating chandeliers; supple leather hugs sweeping benches; responsibly sourced timber blooms into furniture that feels less built than grown. Her palettes—muddy terracotta, river-stone gray, olive green laced with pearlescent accents—whisper of earth and sea.\n\nAt the Luna Sea Residence in Positano, Morelli blurred walls and waves, folding terraces into living rooms with sinuous furniture. Tokyo\'s Nimiko Spa Retreat became a sanctuary of stone and linen. In Oslo\'s Aurora Café and Lounge, fabric draperies ripple overhead like northern lights.\n\nMorelli designs not to dazzle but to nurture. Each room unfolds like a lived story, a stage for quiet connection and emotional resonance.' },
      { id: 'eleanor_reed', name: 'Eleanor Reed', style: 'Vintage Eclectic', bio: 'Mixes vintage patina with contemporary comfort for richly textured, eclectic authenticity.', image: '/images/designers/eleanor_reed.jpg', full_bio: 'Born in the verdant countryside near Asheville, North Carolina, Eleanor Reed\'s bond with art and artisanship formed early. Growing up was synonymous with afternoons spent wandering through antique markets alongside her woodworker father and painter mother.\n\nThe slow rhythms of Appalachian tradition cultivated in her a deeply-rooted reverence for broader Southern crafts—pottery with irregular charm, meticulously hand-loomed fabrics, and cabinetry shaped by hand. Eleanor was captivated by contrasts: age-worn heirlooms reimagined against stark concrete walls, or classic furnishings juxtaposed against bright contemporary artworks.\n\nWith an emerging creativity keenly attuned to tactile beauty, Eleanor pursued her education at the Rhode Island School of Design (RISD). There, Eleanor immersed herself fully in design theory, integrating sculptural forms, architecture fundamentals, sustainable approaches, and historical awareness.\n\nHer philosophy: "imperfection is authenticity." Spaces are collections of stories, each object a chapter that enriches life. She understands that spaces tell stories through their materials, their layers, their honest wear.' },
      { id: 'oliver_renard', name: 'Oliver Renard', style: 'Maximalist Theater', bio: 'Stages maximalist fantasies with jewel-tone palettes, luxe textures, and theatrical storytelling.', image: '/images/designers/oliver_renard.jpg', full_bio: 'Born in Charleston to a set-designer mother and playwright father, Renard spent childhood summers amid folding velvet curtains and hand-painted backdrops. Antique-flocked wallpaper whispered secrets of bygone elegance; ornate gardens rehearsed him in symmetry and surprise.\n\nAt Parsons, Renard fused technical rigor with drama workshops, then crossed the Atlantic to London\'s drama school. Those years taught him to choreograph light and shadow, to script a room\'s emotional arc, to make furniture feel like leading actors.\n\nRenard\'s interiors hum with jewel-toned palettes—emerald booths, ruby drapes, cerulean walls—that pulse against gilded accents and antique heirlooms. Plush velvets collide with lacquered panels; sculptural fixtures hover like stage props. He revels in scale shifts: a low-slung sofa framed by soaring ceiling coffers, a tiny cabaret chair tucked beneath an over-the-top crystal chandelier.\n\nIn Charleston\'s Magnolia Hotel, each guest room unfolds like a chapter: butterfly-patterned murals, brass-bound four-post beds, banquettes draped in opulent silks. At New Orleans\'s Le Carnaval, Renard summoned Mardi Gras exuberance indoors.\n\nOliver Renard doesn\'t just decorate; he directs. His interiors are scripts you inhabit, performances you live.' },
      { id: 'gabrielle_marlowe', name: 'Gabrielle Marlowe', style: 'Southern Refinement', bio: 'Blends Southern graciousness with European classicism to craft airy, refined spaces of quiet luxury.', image: '/images/designers/gabrielle_marlowe.jpg', full_bio: 'Raised among Savannah\'s antebellum portraits and her parents\' historic archives, Gabrielle learned that every cornice and curve carries a story. Childhood summers in Provençal villas and Tuscan palazzos taught her to weave tradition into light-filled interiors.\n\nAt RISD, she mastered spatial choreography—lighting that feels like dawn, palettes drawn from olive groves and sea-sprayed shores, textures that invite the hand. Apprenticeships in New York townhouses and Atlanta\'s high-rise penthouses honed her gift for honoring original architecture while injecting fresh vitality.\n\nHer rooms unfold in soft neutrals—ivory, stone gray, warm cocoa—with judicious notes of deep blue or emerald. She pairs crisp linens and plush velvets with polished woods and brushed metals. Symmetry anchors each vignette; unexpected accents spark intrigue without noise.\n\nBellemeade Estate, Savannah: A Georgian mansion reborn—silken draperies framing antique moldings, deep-blue sofas floating atop reclaimed-wood floors, gilded sconces beside modern abstracts.\n\nGabrielle Marlowe\'s interiors feel both timeless and of-the-moment—a seamless dialogue between past and present.' },
      { id: 'elise_marceau', name: 'Elise Marceau', style: 'Zen Minimalism', bio: 'Balances minimalist restraint with tactile warmth, creating zen-like sanctuaries of European elegance.', image: '/images/designers/elise_marceau.jpg', full_bio: 'Born amid Toulouse\'s medieval streets, she learned texture from her leather-artisan father and composition from her painter mother. Summers wandering Provençal ateliers and Spanish ceramics workshops taught her to see every object as a storyteller.\n\nTrained at École Camondo and refined under Milan\'s luxury ateliers, Marceau balances minimal layouts with sensorial richness: hand-glazed ceramics, softly woven linens, raw oak, and natural stone. Her palette—muted taupes, warm ivories, soft grays—whispers calm, while curves and layered textures invite touch.\n\nChâteau de Lumière, Provence: Original stone arches frame streamlined furnishings in ivory and linen. Custom ceramic pendants by local artisans become luminous focal points; open shelving displays curated travel mementos like gallery vignettes.\n\nTokyo Tranquility Hotel: Twenty private suites merge European restraint with Japanese "Ma"—spaces defined by intentional emptiness. Tactile wall finishes, Kyoto-crafted woodwork, and diffused rice-paper lighting cocoon guests in meditative comfort.\n\nMarceau\'s rooms never shout—yet they linger in memory. She crafts environments that feel both grounded and poetic.' },
      { id: 'alexander_bennett', name: 'Alexander Bennett', style: 'Classical Grandeur', bio: 'Revives classical grandeur with tailored American sophistication and rich architectural detailing.', image: '/images/designers/alexander_bennett.jpg', full_bio: 'Born in Charleston, South Carolina, Alexander Bennett grew up immersed in historical charm and tradition. His father, an eminent architecture historian, specialized in colonial and neoclassical structures, while his mother, an antiques collector, frequented estate sales and gallery openings.\n\nRecognizing his son\'s talent, Bennett\'s father frequently invited Alexander into his lectures, teaching him discernment of key architectural attributes such as molding, symmetry, axial alignment, and proportional clarity. Alexander\'s profound understanding of historical accuracy and classical proportion became evident even as a teenager.\n\nPursuing this passion formally, Bennett studied Architecture and Decorative Arts at the Rhode Island School of Design, followed by advanced studies at Parsons School of Design in New York. During his impactful internship in Paris under renowned architect-designers specializing in classical restoration, Bennett gained vivid insights into rigorous architectural precision.\n\nToday, Alexander Bennett stands distinctly in the field of interior design, shaped by classical architectural precision married seamlessly with sumptuous American-inspired splendor.' },
      { id: 'allegra_marquez', name: 'Allegra Marquez', style: 'Cultural Fusion', bio: 'Combines cultural authenticity with modern lines, marrying vibrant heritage motifs to Scandinavian restraint.', image: '/images/designers/allegra_marquez.jpg', full_bio: 'Born in Valencia, Spain, a city known for its vibrant festivals, dynamic history, magnificent architecture, and pottery traditions. Raised in an artistic family with a ceramicist mother and a father who was an antiques merchant, Allegra grew up surrounded by diverse cultural artifacts and creative traditions.\n\nChildhood holidays traversing through Portugal, Morocco, and Italy further immersed Allegra in the complexities, patterns, and textures of varying cultures and architectural legacies.\n\nDetermined to translate these inspirations professionally, Allegra studied interior architecture at the prestigious Politecnico di Milano. Her rigorous education in Milan was a defining baptism into elegance, craftsmanship, and functional sensibilities informed by Italian modernism.\n\nAllegra\'s signature style quickly matured, characterized by the captivating eclecticism of global craftwork meshed integrally with minimalist modernity—she masterfully balanced vibrant global patterns with a pared-back, mindful simplicity.\n\nSustainable sourcing became integral to her philosophy; Allegra insisted on ethical provenance, believing interiors ought to speak not merely of sophistication but of integrity, mindful consumption, and respect for global cultures.' },
      { id: 'olivia_bennett', name: 'Olivia Bennett', style: 'Approachable Elegance', bio: 'Creates approachable elegance through thoughtful styling and sustainable, handcrafted details.', image: '/images/designers/olivia_bennett.jpg', full_bio: 'Olivia Bennett was born on a gentle spring morning in the picturesque countryside near Fredericton, New Brunswick. Growing up as the youngest of four siblings in a cozy, century-old farmhouse, Olivia\'s formative years were steeped in creative exploration, family warmth, and a genuine love for nature and heritage spaces.\n\nHer mother, a local artist, often encouraged playful creativity, allowing Olivia\'s inherent sensibility for color and aesthetics to emerge naturally.\n\nFollowing secondary school, Olivia\'s affinity for the interplay of space, texture, and color drove her to pursue studies at Ryerson University\'s acclaimed School of Interior Design in Toronto. During her studies, Olivia quickly established herself as a promising talent.\n\nAfter graduation, Olivia interned with notable designers in both Toronto and New York, eventually working alongside senior designers on notable boutique hotel redevelopments.\n\nSpurred by an urge to carve her own path, Olivia established her studio "Bennett Home Interiors" in Prince Edward County, Ontario. The peaceful community, surrounded by rural charm, vineyards, and water views, inspired her to nurture a design approach rooted in relaxed elegance and livable beauty.' }
    ];

    const colorSchemes = [
      { id: 'glacial_muse', name: 'Glacial Muse', colors: ['#E8F4F8', '#B8D8E8', '#89B5CE'], description: 'Icy pastels and frosted neutrals evoking Nordic serenity' },
      { id: 'nomad_prism', name: 'Nomad Prism', colors: ['#D4A574', '#8B7355', '#E6D5C3'], description: 'Vibrant gems and wanderlust tones for eclectic tastes' },
      { id: 'urban_alloy', name: 'Urban Alloy', colors: ['#4A4A4A', '#7D7D7D', '#A8A8A8'], description: 'Iron hues and industrial patina—gritty and raw' },
      { id: 'aegean_whisper', name: 'Aegean Whisper', colors: ['#5B9AA9', '#D4C5A9', '#E8DCC4'], description: 'Oceanic blues and sun-kissed earth tones' },
      { id: 'velvet_deco', name: 'Velvet Deco', colors: ['#4A1E3D', '#8B6F47', '#C9A961'], description: 'Deep jewel tones and metallic glamour' },
      { id: 'desert_modern', name: 'Desert Modern', colors: ['#C79F6B', '#8D6346', '#E8DCC4'], description: 'Burnt earth and washed neutrals with desert grace' },
      { id: 'enchanted_forest', name: 'Enchanted Forest', colors: ['#2D5016', '#4A7C2F', '#8B9E6B'], description: 'Lush emeralds and bark browns meet mossy whispers' },
      { id: 'savannah_bloom', name: 'Savannah Bloom', colors: ['#D4A960', '#B8935F', '#E8C98E'], description: 'Sunburnt petals and golden grass' },
      { id: 'canyon_clay', name: 'Canyon Clay', colors: ['#B85C3F', '#8D5241', '#D4A07A'], description: 'Terracotta cliffs under a molten sky' },
      { id: 'lunar_drift', name: 'Lunar Drift', colors: ['#B8B8C8', '#9494A8', '#D4D4E0'], description: 'Icy greys, pale lavenders, and shadows in motion' },
      { id: 'sienna_smoke', name: 'Sienna Smoke', colors: ['#A67C52', '#8D6E5A', '#C9B4A0'], description: 'Warm neutrals drifting through dusty clay and chalk' },
      { id: 'retro_zest', name: 'Retro Zest', colors: ['#8B9E4A', '#E89B4F', '#E8D960'], description: 'Avocado green, popsicle orange, lemony optimism' },
      { id: 'twilight_grove', name: 'Twilight Grove', colors: ['#6B5B7C', '#4A5941', '#8D8E9E'], description: 'Smoky violet, ash green, and forest shadows' },
      { id: 'citrus_pop', name: 'Citrus Pop', colors: ['#E89B4F', '#E8D960', '#FFB84D'], description: 'Grapefruit zest and neon fizz—sunrise energy' },
      { id: 'oxblood_study', name: 'Oxblood Study', colors: ['#5B1E1E', '#3D2929', '#8D6B6B'], description: 'Oxblood, ink, and old paper tones—academic luxury' },
      { id: 'sunken_studio', name: 'Sunken Studio', colors: ['#3D4A5B', '#2D3D4F', '#6B7C8D'], description: 'Undersea study in moody ink and shale' },
      { id: 'charred_cotton', name: 'Charred Cotton', colors: ['#5B5B5B', '#8D8D8D', '#C9C9C9'], description: 'Ash, linen, and charcoal smudge' },
      { id: 'silken_ember', name: 'Silken Ember', colors: ['#A66B5F', '#8D5747', '#C9A89E'], description: 'Firelight meets silk—subdued luxury with spice' },
      { id: 'mineral_tonic', name: 'Mineral Tonic', colors: ['#5B7C8D', '#7C8E9E', '#A0B4C0'], description: 'Mineral blue, flint, and dried herbs' },
      { id: 'bauhaus_dusk', name: 'Bauhaus Dusk', colors: ['#8D8E9E', '#4A5B7C', '#C9A961'], description: 'Modernist primary accents on greys and pastels' }
    ];

    return (
      <div className="space-y-8">
        {/* Foundation Lock Modal */}
        {showFoundationModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4" onClick={() => setShowFoundationModal(false)}>
            <div className="bg-white rounded-lg max-w-2xl w-full" onClick={(e) => e.stopPropagation()}>
              <div className="p-8">
                <div className="flex items-start justify-between mb-6">
                  <div className="flex items-center space-x-4">
                    <div className="text-5xl">🔒</div>
                    <div>
                      <h2 className="text-2xl font-bold text-gray-900">Foundation Content Required</h2>
                      <p className="text-sm text-gray-600 mt-1">These tools are locked until foundation content is ready</p>
                    </div>
                  </div>
                  <button 
                    onClick={() => setShowFoundationModal(false)}
                    className="text-gray-400 hover:text-gray-600 text-2xl"
                  >
                    ×
                  </button>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
                  <h3 className="font-semibold text-blue-900 mb-3">What are Foundation Modules?</h3>
                  <p className="text-blue-800 mb-4">
                    When you create a listing, we automatically generate 3 essential pieces of content that form the "foundation":
                  </p>
                  <ul className="space-y-2 text-blue-800">
                    <li className="flex items-start">
                      <span className="mr-2">📍</span>
                      <span><strong>Neighborhood Research</strong> - Local amenities, schools, transportation, community vibe</span>
                    </li>
                    <li className="flex items-start">
                      <span className="mr-2">📝</span>
                      <span><strong>Property Description</strong> - Compelling listing copy highlighting key features</span>
                    </li>
                    <li className="flex items-start">
                      <span className="mr-2">📊</span>
                      <span><strong>Market Intelligence</strong> - Target buyers, positioning, pricing strategy</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mb-6">
                  <h3 className="font-semibold text-yellow-900 mb-3">Why are other tools locked?</h3>
                  <p className="text-yellow-800 mb-3">
                    Other AI tools (marketing copy, social posts, emails, buyer profiles) need the foundation content to create relevant, contextual results.
                  </p>
                  <p className="text-yellow-800">
                    <strong>Foundation Status:</strong> {listing?.foundation_status === 'processing' ? '⏳ Processing...' : listing?.foundation_status === 'completed' ? '✅ Complete' : '❌ Not started'}
                  </p>
                </div>

                <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-6">
                  <h3 className="font-semibold text-green-900 mb-3">What to do now?</h3>
                  {listing?.foundation_status === 'processing' ? (
                    <div className="text-green-800">
                      <p className="mb-2">✅ Foundation content is currently being generated (usually takes 30-60 seconds)</p>
                      <p className="mb-2">✅ This page will auto-refresh every 5 seconds</p>
                      <p>✅ Once complete, all tools will unlock automatically</p>
                    </div>
                  ) : listing?.foundation_status === 'completed' ? (
                    <div className="text-green-800">
                      <p className="mb-2">✅ Foundation content is complete!</p>
                      <p>✅ Close this modal and try clicking the tool again</p>
                    </div>
                  ) : (
                    <div className="text-green-800">
                      <p className="mb-2">The foundation content hasn't been generated yet.</p>
                      <p>Please refresh the page or contact support if this persists.</p>
                    </div>
                  )}
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => setShowFoundationModal(false)}
                    className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors font-semibold"
                  >
                    Got it!
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Designer Bio Modal */}
        {showDesignerModal && selectedDesigner && (
          <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4" onClick={() => setShowDesignerModal(false)}>
            <div className="bg-white rounded-lg max-w-3xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="p-8">
                <div className="flex items-start space-x-6 mb-6">
                  <img 
                    src={selectedDesigner.image} 
                    alt={selectedDesigner.name}
                    className="w-32 h-32 object-cover rounded-lg flex-shrink-0"
                    onError={(e) => {
                      e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128"><rect fill="%23e5e7eb" width="128" height="128"/><text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="%239ca3af" font-size="48">👤</text></svg>';
                    }}
                  />
                  <div className="flex-1">
                    <h2 className="text-3xl font-bold text-gray-900 mb-2">{selectedDesigner.name}</h2>
                    <div className="text-lg font-semibold text-blue-600 mb-4">{selectedDesigner.style}</div>
                  </div>
                  <button 
                    onClick={() => setShowDesignerModal(false)}
                    className="text-gray-400 hover:text-gray-600 text-2xl"
                  >
                    ×
                  </button>
                </div>
                <div className="prose max-w-none">
                  <p className="text-gray-700 whitespace-pre-line leading-relaxed">
                    {selectedDesigner.full_bio}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* AI Interior Design Section */}
        <div className="border-t border-gray-200 pt-6">
            {/* Info Banner */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <p className="text-sm text-blue-800">
                🎨 <strong>AI Interior Design:</strong> Choose designer and colors on the left, then upload and select images on the right. Process with AI (5 credits per image)
              </p>
            </div>

            {/* 2-Column Layout: Designers/Colors Left, Photos Right */}
            <div className="grid lg:grid-cols-2 gap-8">
              {/* LEFT COLUMN - Designers & Colors */}
              <div className="space-y-8">
                {/* Designer Gallery */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Choose Your Designer</h2>
              <p className="text-gray-600 mb-4">Click any designer card to select their style</p>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {[
                  { id: 'alessia_duval', name: 'Alessia Duval', style: 'Parisian Eclectic', bio: 'Infuses Parisian elegance with global eclecticism', image: '/images/designers/alessia_duval.jpg' },
                  { id: 'adrian_mercer', name: 'Adrian Mercer', style: 'Industrial Poetry', bio: 'Transforms post-industrial materials into poetic interiors', image: '/images/designers/adrian_mercer.jpg' },
                  { id: 'lucien_hart', name: 'Lucien Hart', style: 'Couture Glamour', bio: 'Fuses runway glamour with architectural audacity', image: '/images/designers/lucien_hart.jpg' },
                  { id: 'elinor_hartwell', name: 'Elinor Hartwell', style: 'Mindful Comfort', bio: 'Creates warm, soulful rooms with tactile comfort', image: '/images/designers/elinor_hartwell.jpg' },
                  { id: 'bianca_morelli', name: 'Bianca Morelli', style: 'Organic Elegance', bio: 'Weaves fluid, organic forms into elegant interiors', image: '/images/designers/bianca_morelli.jpg' },
                  { id: 'eleanor_reed', name: 'Eleanor Reed', style: 'Vintage Eclectic', bio: 'Mixes vintage patina with contemporary comfort', image: '/images/designers/eleanor_reed.jpg' },
                  { id: 'oliver_renard', name: 'Oliver Renard', style: 'Maximalist Theater', bio: 'Stages maximalist fantasies with jewel-tone palettes', image: '/images/designers/oliver_renard.jpg' },
                  { id: 'gabrielle_marlowe', name: 'Gabrielle Marlowe', style: 'Southern Refinement', bio: 'Blends Southern graciousness with European classicism', image: '/images/designers/gabrielle_marlowe.jpg' },
                  { id: 'elise_marceau', name: 'Elise Marceau', style: 'Zen Minimalism', bio: 'Balances minimalist restraint with tactile warmth', image: '/images/designers/elise_marceau.jpg' },
                  { id: 'alexander_bennett', name: 'Alexander Bennett', style: 'Classical Grandeur', bio: 'Revives classical grandeur with American sophistication', image: '/images/designers/alexander_bennett.jpg' },
                  { id: 'allegra_marquez', name: 'Allegra Marquez', style: 'Cultural Fusion', bio: 'Combines cultural authenticity with modern lines', image: '/images/designers/allegra_marquez.jpg' },
                  { id: 'olivia_bennett', name: 'Olivia Bennett', style: 'Approachable Elegance', bio: 'Creates approachable elegance through thoughtful styling', image: '/images/designers/olivia_bennett.jpg' }
                ].map(designer => (
                  <div 
                    key={designer.id} 
                    className={`bg-white border-2 rounded-lg overflow-hidden hover:shadow-lg transition-all cursor-pointer ${
                      globalDesigner === designer.id ? 'border-blue-600 shadow-lg ring-2 ring-blue-400' : 'border-gray-200 hover:border-blue-400'
                    }`}
                    onClick={() => setGlobalDesigner(designer.id)}
                  >
                    <div className="h-40 overflow-hidden">
                      <img 
                        src={designer.image} 
                        alt={designer.name}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><rect fill="%23e5e7eb" width="200" height="200"/><text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="%239ca3af" font-size="60">👤</text></svg>';
                        }}
                      />
                    </div>
                    <div className="p-3">
                      <h3 className="font-bold text-gray-900 text-sm">{designer.name}</h3>
                      <div className="text-xs font-semibold text-blue-600 mb-1">{designer.style}</div>
                      <p className="text-xs text-gray-600 line-clamp-2">{designer.bio}</p>
                    </div>
                    {globalDesigner === designer.id && (
                      <div className="bg-blue-600 text-white text-center py-1 text-xs font-semibold">
                        ✓ Selected
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Color Schemes Gallery */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Choose Color Scheme</h2>
              <p className="text-gray-600 mb-4">Click any palette to select</p>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {[
                  { id: 'glacial_muse', name: 'Glacial Muse', colors: ['#E8F4F8', '#B8D8E8', '#89B5CE'], description: 'Icy pastels and frosted neutrals' },
                  { id: 'nomad_prism', name: 'Nomad Prism', colors: ['#D4A574', '#8B7355', '#E6D5C3'], description: 'Vibrant gems and wanderlust tones' },
                  { id: 'urban_alloy', name: 'Urban Alloy', colors: ['#4A4A4A', '#7D7D7D', '#A8A8A8'], description: 'Iron hues and industrial patina' },
                  { id: 'aegean_whisper', name: 'Aegean Whisper', colors: ['#5B9AA9', '#D4C5A9', '#E8DCC4'], description: 'Oceanic blues and sun-kissed earth' },
                  { id: 'velvet_deco', name: 'Velvet Deco', colors: ['#4A1E3D', '#8B6F47', '#C9A961'], description: 'Deep jewel tones and metallic glamour' },
                  { id: 'desert_modern', name: 'Desert Modern', colors: ['#C79F6B', '#8D6346', '#E8DCC4'], description: 'Burnt earth and washed neutrals' },
                  { id: 'enchanted_forest', name: 'Enchanted Forest', colors: ['#2D5016', '#4A7C2F', '#8B9E6B'], description: 'Lush emeralds and bark browns' },
                  { id: 'savannah_bloom', name: 'Savannah Bloom', colors: ['#D4A960', '#B8935F', '#E8C98E'], description: 'Sunburnt petals and golden grass' },
                  { id: 'canyon_clay', name: 'Canyon Clay', colors: ['#B85C3F', '#8D5241', '#D4A07A'], description: 'Terracotta cliffs under molten sky' },
                  { id: 'lunar_drift', name: 'Lunar Drift', colors: ['#B8B8C8', '#9494A8', '#D4D4E0'], description: 'Icy greys and pale lavenders' },
                  { id: 'sienna_smoke', name: 'Sienna Smoke', colors: ['#A67C52', '#8D6E5A', '#C9B4A0'], description: 'Warm neutrals and dusty clay' },
                  { id: 'retro_zest', name: 'Retro Zest', colors: ['#8B9E4A', '#E89B4F', '#E8D960'], description: 'Avocado green and popsicle orange' },
                  { id: 'twilight_grove', name: 'Twilight Grove', colors: ['#6B5B7C', '#4A5941', '#8D8E9E'], description: 'Smoky violet and forest shadows' },
                  { id: 'citrus_pop', name: 'Citrus Pop', colors: ['#E89B4F', '#E8D960', '#FFB84D'], description: 'Grapefruit zest and neon fizz' },
                  { id: 'oxblood_study', name: 'Oxblood Study', colors: ['#5B1E1E', '#3D2929', '#8D6B6B'], description: 'Oxblood and old paper tones' },
                  { id: 'sunken_studio', name: 'Sunken Studio', colors: ['#3D4A5B', '#2D3D4F', '#6B7C8D'], description: 'Undersea study in moody ink' },
                  { id: 'charred_cotton', name: 'Charred Cotton', colors: ['#5B5B5B', '#8D8D8D', '#C9C9C9'], description: 'Ash, linen, and charcoal' },
                  { id: 'silken_ember', name: 'Silken Ember', colors: ['#A66B5F', '#8D5747', '#C9A89E'], description: 'Firelight meets silk with spice' },
                  { id: 'mineral_tonic', name: 'Mineral Tonic', colors: ['#5B7C8D', '#7C8E9E', '#A0B4C0'], description: 'Mineral blue and dried herbs' },
                  { id: 'bauhaus_dusk', name: 'Bauhaus Dusk', colors: ['#8D8E9E', '#4A5B7C', '#C9A961'], description: 'Modernist accents on pastels' }
                ].map(scheme => (
                  <div 
                    key={scheme.id} 
                    className={`bg-white border-2 rounded-lg p-3 hover:shadow-lg transition-all cursor-pointer ${
                      globalColorScheme === scheme.id ? 'border-blue-600 shadow-lg ring-2 ring-blue-400' : 'border-gray-200 hover:border-blue-400'
                    }`}
                    onClick={() => setGlobalColorScheme(scheme.id)}
                  >
                    <div className="flex space-x-1 mb-2">
                      {scheme.colors.map((color, i) => (
                        <div
                          key={i}
                          className="flex-1 h-12 rounded"
                          style={{ backgroundColor: color }}
                        />
                      ))}
                    </div>
                    <h3 className="font-semibold text-gray-900 text-sm mb-1">{scheme.name}</h3>
                    <p className="text-xs text-gray-600">{scheme.description}</p>
                    {globalColorScheme === scheme.id && (
                      <div className="mt-2 bg-blue-600 text-white text-center py-1 rounded text-xs font-semibold">
                        ✓ Selected
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Optional Custom Overrides */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Custom Options (Optional)</h3>
              <p className="text-sm text-gray-600 mb-4">Override designer and colors with your own preferences</p>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Custom Design Description
                  </label>
                  <textarea
                    value={customDescription}
                    onChange={(e) => setCustomDescription(e.target.value)}
                    placeholder="e.g., Modern coastal vibes with natural textures..."
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg text-sm resize-none"
                    rows="2"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Custom Colors
                  </label>
                  <input
                    type="text"
                    value={customColors}
                    onChange={(e) => setCustomColors(e.target.value)}
                    placeholder="e.g., Soft sage green, warm beige, ivory white..."
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg text-sm"
                  />
                </div>
              </div>
            </div>
              </div>
              {/* End Left Column */}
              
              {/* RIGHT COLUMN - Photos & Upload */}
              <div className="space-y-6">
                {/* Upload Section */}
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
                    <span className="text-sm text-gray-600">Select multiple images at once</span>
                  </div>
                </div>

                {/* Action Header */}
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Select Images & Set Room Types ({selectedImages.length} selected)
                  </h3>
              <button
                onClick={handleProcessInteriorDesign}
                disabled={selectedImages.length === 0 || processingDesign}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 font-semibold"
              >
                {processingDesign ? 'Processing...' : `Process ${selectedImages.length} Image${selectedImages.length !== 1 ? 's' : ''} (${selectedImages.length * 5} credits)`}
              </button>
            </div>

            {/* Simplified Image Grid - Only Room Type Per Image */}
            <div className="grid md:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
              {images.map(image => {
                const imageSettings = selectedImages.find(s => s.image_id === image.id);
                const isSelected = !!imageSettings;

                return (
                  <div
                    key={image.id}
                    className={`border-2 rounded-lg overflow-hidden transition-all cursor-pointer ${
                      isSelected
                        ? 'border-blue-600 shadow-lg'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => {
                      if (!isSelected) {
                        setSelectedImages(prev => [...prev, {
                          image_id: image.id,
                          room_type: 'living_room'
                        }]);
                      }
                    }}
                  >
                    {/* Image */}
                    <div className="relative">
                      <img
                        src={`${BACKEND_URL}${image.url}`}
                        alt={image.filename}
                        className="w-full h-40 object-cover"
                      />
                      {isSelected && (
                        <div className="absolute top-2 right-2 bg-blue-600 text-white px-2 py-1 rounded-full font-semibold text-xs">
                          ✓
                        </div>
                      )}
                    </div>

                    {/* Room Type Selection */}
                    {isSelected && (
                      <div className="p-3 bg-blue-50 space-y-2">
                        <select
                          value={imageSettings.room_type}
                          onChange={(e) => {
                            e.stopPropagation();
                            setSelectedImages(prev =>
                              prev.map(s => s.image_id === image.id 
                                ? {...s, room_type: e.target.value}
                                : s
                              )
                            );
                          }}
                          className="w-full px-2 py-2 border border-gray-300 rounded text-xs"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <option value="living_room">Living Room</option>
                          <option value="bedroom">Bedroom</option>
                          <option value="kitchen">Kitchen</option>
                          <option value="bathroom">Bathroom</option>
                          <option value="dining_room">Dining Room</option>
                          <option value="office">Office</option>
                          <option value="exterior">Exterior</option>
                        </select>
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedImages(prev => prev.filter(s => s.image_id !== image.id));
                          }}
                          className="w-full py-1 bg-red-100 text-red-700 hover:bg-red-200 rounded text-xs font-medium transition-colors"
                        >
                          Remove
                        </button>
                      </div>
                    )}
                    
                    {!isSelected && (
                      <div className="p-3 bg-white text-center">
                        <span className="text-xs text-gray-500">Click to select</span>
                      </div>
                    )}
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
                      <div className="p-3 bg-white space-y-2">
                        <div className="text-xs text-gray-600">
                          {variant.room_type} • {variant.designer}
                        </div>
                        <div className="text-xs text-gray-500">
                          {variant.color_scheme}
                        </div>
                        <div className="text-xs text-gray-500">
                          {variant.status === 'completed' ? '✓ Completed' : '⏳ Processing...'}
                        </div>
                        
                        {variant.status === 'completed' && variant.processed_image_url && (
                          <div className="flex space-x-2 pt-2">
                            <a
                              href={`${BACKEND_URL}${variant.processed_image_url}`}
                              download={`interior-design-${variant.room_type}-${variant.designer}.jpg`}
                              className="flex-1 bg-green-600 text-white text-xs py-2 px-3 rounded hover:bg-green-700 transition-colors text-center"
                            >
                              📥 Download
                            </a>
                            <button
                              onClick={() => handleRerunDesign(variant)}
                              className="flex-1 bg-blue-600 text-white text-xs py-2 px-3 rounded hover:bg-blue-700 transition-colors"
                            >
                              🔄 Rerun
                            </button>
                          </div>
                        )}
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
          {/* End Right Column */}
        </div>
        {/* End 2-Column Grid */}
      </div>
      {/* End AI Interior Design Section */}
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
    
    const { property_details } = listing;
    const approvedDesigns = listing.interior_design_variants?.filter(v => v.status === 'completed') || [];
    const heroImage = approvedDesigns[0]?.processed_image_url || (images[0]?.url);
    
    // Use geocoded coordinates if available, otherwise use stored coordinates or default
    const latitude = mapCoordinates?.lat || property_details.latitude || 39.8283;
    const longitude = mapCoordinates?.lng || property_details.longitude || -98.5795;
    
    // Define foundation status variables
    const foundationComplete = listing.foundation_status === 'completed';
    const foundationProcessing = listing.foundation_status === 'processing';
    const foundationModules = modules.filter(m => m.isFoundation);
    const dependentModules = modules.filter(m => m.requiresFoundation);
    const otherModules = modules.filter(m => !m.isFoundation && !m.requiresFoundation && !m.ai);

    return (
      <div className="space-y-6">
        {/* Hero Image */}
        {heroImage && (
          <div className="relative h-96 rounded-lg overflow-hidden shadow-lg">
            <img
              src={`${BACKEND_URL}${heroImage}`}
              alt={property_details.address}
              className="w-full h-full object-cover"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-6">
              <h1 className="text-4xl font-bold text-white mb-2">
                {property_details.address}
              </h1>
              <p className="text-xl text-white/90">
                {property_details.city}, {property_details.state} {property_details.zip_code}
              </p>
            </div>
            {property_details.listing_price && (
              <div className="absolute top-6 right-6 bg-blue-600 text-white px-6 py-3 rounded-lg shadow-lg">
                <div className="text-sm opacity-90">Listed at</div>
                <div className="text-2xl font-bold">${property_details.listing_price.toLocaleString()}</div>
              </div>
            )}
          </div>
        )}

        {/* Map Section */}
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-2xl font-bold text-gray-900">Location</h2>
            <p className="text-gray-600 mt-1">
              {property_details.address}, {property_details.city}, {property_details.state}
            </p>
            {geocodingAddress && (
              <p className="text-sm text-blue-600 mt-1">📍 Finding exact location...</p>
            )}
          </div>
          <div style={{ height: '400px', width: '100%' }} className="relative">
            {!mapCoordinates && geocodingAddress ? (
              <div className="flex items-center justify-center h-full bg-gray-100">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Loading map...</p>
                </div>
              </div>
            ) : (
              <MapContainer
                key={`map-${listing.id}-${latitude}-${longitude}`}
                center={[latitude, longitude]}
                zoom={15}
                style={{ height: '100%', width: '100%', zIndex: 1 }}
                scrollWheelZoom={false}
                whenCreated={(map) => {
                  setTimeout(() => {
                    map.invalidateSize();
                  }, 100);
                }}
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <Marker position={[latitude, longitude]}>
                  <Popup>
                    <div className="text-center">
                      <strong>{property_details.address}</strong>
                      <br />
                      {property_details.city}, {property_details.state}
                    </div>
                  </Popup>
                </Marker>
              </MapContainer>
            )}
          </div>
          
          {/* Future API Integration Placeholders */}
          <div className="p-6 bg-gray-50 border-t border-gray-200">
            <div className="grid md:grid-cols-4 gap-4 text-center">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <div className="text-2xl mb-1">🏫</div>
                <div className="text-sm text-gray-600">School Ratings</div>
                <div className="text-xs text-gray-400 mt-1">Coming soon</div>
              </div>
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <div className="text-2xl mb-1">🚶</div>
                <div className="text-sm text-gray-600">Walk Score</div>
                <div className="text-xs text-gray-400 mt-1">Coming soon</div>
              </div>
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <div className="text-2xl mb-1">🚇</div>
                <div className="text-sm text-gray-600">Transit Score</div>
                <div className="text-xs text-gray-400 mt-1">Coming soon</div>
              </div>
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <div className="text-2xl mb-1">🛡️</div>
                <div className="text-sm text-gray-600">Safety Rating</div>
                <div className="text-xs text-gray-400 mt-1">Coming soon</div>
              </div>
            </div>
          </div>
        </div>

        {/* Foundation Processing Indicator */}
        {listing.foundation_status === 'processing' && (
          <div className="bg-blue-50 border-2 border-blue-300 rounded-lg shadow-lg p-6 animate-pulse">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-bold text-gray-900 mb-2">🔄 Generating Your Foundation Content...</h3>
                <p className="text-gray-700 mb-3">Our AI is creating professional content for your listing. This typically takes 1-2 minutes.</p>
                <div className="bg-white rounded-lg p-3 border border-blue-200">
                  <div className="text-sm font-medium text-gray-700 mb-2">What's being created:</div>
                  <div className="space-y-1 text-sm text-gray-600">
                    <div>✓ Neighborhood Research (schools, amenities, demographics)</div>
                    <div>✓ Professional Property Description (listing copy)</div>
                    <div>✓ Market Intelligence (buyer targeting & positioning)</div>
                  </div>
                </div>
                <div className="mt-3 text-sm text-gray-600">
                  💡 This page will auto-refresh when complete. Feel free to wait or come back in a minute!
                </div>
              </div>
            </div>
          </div>
        )}

        {/* What's Next Guidance - Show after foundation completes */}
        {listing.foundation_status === 'completed' && (
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg shadow-lg p-6">
            <div className="flex items-start space-x-4">
              <div className="bg-green-500 rounded-full p-3 flex-shrink-0">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-bold text-gray-900 mb-2">🎉 Foundation Complete!</h3>
                <p className="text-gray-700 mb-4">Your property foundation content is ready. Here's what you can do next:</p>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white rounded-lg p-4 border border-green-200">
                    <div className="flex items-center space-x-3 mb-2">
                      <span className="text-2xl">📸</span>
                      <div>
                        <div className="font-semibold text-gray-900">Upload Photos</div>
                        <div className="text-sm text-gray-600">5 credits per image</div>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-3">Transform property photos with AI interior design</p>
                    <button
                      onClick={() => { setActiveModule('images'); setActiveView('module'); }}
                      className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition-colors text-sm font-medium"
                    >
                      Go to Photos
                    </button>
                  </div>

                  <div className="bg-white rounded-lg p-4 border border-green-200">
                    <div className="flex items-center space-x-3 mb-2">
                      <span className="text-2xl">📝</span>
                      <div>
                        <div className="font-semibold text-gray-900">Marketing Content</div>
                        <div className="text-sm text-gray-600">Varies by tool</div>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-3">Generate social posts, emails, and more</p>
                    <div className="text-sm text-gray-500">← Select tools from sidebar</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Property Details Grid */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Property Details</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Property Type</div>
              <div className="text-lg font-semibold text-gray-900">{property_details.property_type}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Bedrooms</div>
              <div className="text-lg font-semibold text-gray-900">{property_details.beds} beds</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Bathrooms</div>
              <div className="text-lg font-semibold text-gray-900">{property_details.baths} baths</div>
            </div>
            {property_details.sqft && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600 mb-1">Square Feet</div>
                <div className="text-lg font-semibold text-gray-900">{property_details.sqft.toLocaleString()} sq ft</div>
              </div>
            )}
            {property_details.listing_price && (
              <div className="bg-blue-50 p-4 rounded-lg border-2 border-blue-200">
                <div className="text-sm text-blue-600 mb-1">Listing Price</div>
                <div className="text-2xl font-bold text-blue-600">${property_details.listing_price.toLocaleString()}</div>
              </div>
            )}
          </div>
        </div>

        {/* Approved Interior Designs */}
        {approvedDesigns.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">AI Interior Designs</h2>
              <span className="bg-green-100 text-green-800 text-sm px-3 py-1 rounded-full font-medium">
                {approvedDesigns.length} Approved
              </span>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              {approvedDesigns.map(design => (
                <div key={design.id} className="group relative">
                  <div className="aspect-w-16 aspect-h-12 rounded-lg overflow-hidden shadow-md">
                    <img
                      src={`${BACKEND_URL}${design.processed_image_url}`}
                      alt={`${design.room_type} - ${design.designer}`}
                      className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  </div>
                  <div className="mt-3">
                    <div className="text-sm font-semibold text-gray-900">{design.room_type.replace('_', ' ')}</div>
                    <div className="text-xs text-gray-600">{design.designer.replace('_', ' ')} • {design.color_scheme.replace('_', ' ')}</div>
                  </div>
                  <div className="flex space-x-2 mt-2">
                    <a
                      href={`${BACKEND_URL}${design.processed_image_url}`}
                      download={`interior-design-${design.room_type}.jpg`}
                      className="flex-1 bg-blue-600 text-white text-xs py-2 px-3 rounded hover:bg-blue-700 transition-colors text-center"
                    >
                      📥 Download
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Description */}
        {listing.description && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Description</h2>
            <p className="text-gray-700 leading-relaxed whitespace-pre-line">{listing.description}</p>
          </div>
        )}

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
                    if (isLocked) {
                      setShowFoundationModal(true);
                    } else {
                      setActiveModule(module.id);
                      setActiveView('module');
                    }
                  }}
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
              renderOverview()
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
      
      {/* Module Questionnaire Modal */}
      <ModuleQuestionnaireModal
        isOpen={showQuestionnaireModal}
        onClose={() => setShowQuestionnaireModal(false)}
        moduleName={questionnaireModule}
        writingStyles={writingStyles}
        onSubmit={(formInputs) => handleGenerateContent(questionnaireModule, formInputs)}
        isGenerating={generatingModule === questionnaireModule}
      />
    </div>
  );
};

export default IndividualListingPage;
