import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import AuthModal from './components/auth/AuthModal';
import ProtectedRoute from './components/auth/ProtectedRoute';
import UserDashboard from './components/dashboard/UserDashboard';
import LandingPage from './components/LandingPage';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Header Component
const Header = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);

  return (
    <header className="bg-gradient-to-r from-blue-900 to-blue-700 text-white shadow-lg">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-400 rounded-lg flex items-center justify-center">
              <span className="text-xl font-bold">PA</span>
            </div>
            <h1 className="text-2xl font-bold">ProAgentTools</h1>
          </div>
          
          <div className="flex items-center space-x-6">
            <nav className="hidden md:flex space-x-6">
              <a href="#interior-design" className="hover:text-blue-300 transition-colors">Interior Design</a>
              <a href="#ai-tools" className="hover:text-blue-300 transition-colors">AI Tools</a>
              {isAuthenticated ? (
                <button
                  onClick={() => window.location.href = '/dashboard'}
                  className="hover:text-blue-300 transition-colors"
                >
                  Dashboard
                </button>
              ) : (
                <a href="#pricing" className="hover:text-blue-300 transition-colors">Pricing</a>
              )}
            </nav>
            
            <div className="flex items-center space-x-4">
              {isAuthenticated ? (
                <div className="flex items-center space-x-4">
                  <div className="text-sm">
                    <div className="font-medium">{user?.full_name}</div>
                    <div className="text-blue-200">{user?.credits} credits</div>
                  </div>
                  <button
                    onClick={logout}
                    className="text-blue-200 hover:text-white transition-colors text-sm"
                  >
                    Sign Out
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="bg-blue-500 hover:bg-blue-400 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                >
                  Sign In
                </button>
              )}
            </div>
          </div>
        </div>
        
        {showAuthModal && (
          <AuthModal 
            isOpen={showAuthModal} 
            onClose={() => setShowAuthModal(false)} 
          />
        )}
      </div>
    </header>
  );
};

// Hero Section Component
const HeroSection = () => (
  <section className="relative bg-gradient-to-br from-blue-50 to-indigo-100 py-20">
    <div className="absolute inset-0 bg-black bg-opacity-10"></div>
    <div 
      className="absolute inset-0 bg-cover bg-center bg-no-repeat opacity-20"
      style={{
        backgroundImage: `url('https://images.unsplash.com/photo-1532495142380-2f10c263c93d?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1Nzd8MHwxfHNlYXJjaHwzfHxyZWFsJTIwZXN0YXRlfGVufDB8fHxibHVlfDE3NTMwMjcxNzl8MA&ixlib=rb-4.1.0&q=85')`
      }}
    ></div>
    <div className="relative container mx-auto px-6 text-center">
      <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
        AI-Powered Tools for <span className="text-blue-600">Real Estate Agents</span>
      </h1>
      <p className="text-xl text-gray-700 mb-8 max-w-3xl mx-auto">
        Transform your real estate business with cutting-edge AI technology. 
        Generate stunning interior designs, create compelling property descriptions, 
        and access 30+ professional AI tools designed specifically for agents.
      </p>
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <a href="#interior-design" className="bg-blue-600 text-white px-8 py-4 rounded-lg font-semibold hover:bg-blue-700 transition-colors">
          Try Interior Design AI
        </a>
        <a href="#ai-tools" className="bg-white text-blue-600 border-2 border-blue-600 px-8 py-4 rounded-lg font-semibold hover:bg-blue-50 transition-colors">
          Explore AI Tools
        </a>
      </div>
    </div>
  </section>
);

// Interior Design Tool Component
const InteriorDesignTool = () => {
  const [processing, setProcessing] = useState(false);
  const [processedImage, setProcessedImage] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [currentJobId, setCurrentJobId] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  
  // Queue and history state
  const [activeJobs, setActiveJobs] = useState([]);
  const [jobHistory, setJobHistory] = useState([]);
  const [queueStatus, setQueueStatus] = useState({ queued: 0, processing: 0 });
  
  // Image viewer modal state
  const [viewingImage, setViewingImage] = useState(null);
  
  // New state for design preferences
  const [selectedRoomType, setSelectedRoomType] = useState('living_room');
  const [selectedDesigner, setSelectedDesigner] = useState('alessia_duval');
  const [selectedColorScheme, setSelectedColorScheme] = useState('glacial_muse');
  
  // Configuration data
  const [roomTypes, setRoomTypes] = useState([]);
  const [designers, setDesigners] = useState([]);
  const [colorSchemes, setColorSchemes] = useState([]);
  
  // Modal state for designer bio
  const [showDesignerModal, setShowDesignerModal] = useState(false);
  const [selectedDesignerForModal, setSelectedDesignerForModal] = useState(null);
  
  // Load configuration data on component mount
  useEffect(() => {
    const loadConfigData = async () => {
      try {
        const [roomTypesRes, designersRes, colorSchemesRes] = await Promise.all([
          axios.get(`${API}/interior-design/room-types`),
          axios.get(`${API}/interior-design/designers`),
          axios.get(`${API}/interior-design/color-schemes`)
        ]);
        
        setRoomTypes(roomTypesRes.data.room_types);
        setDesigners(designersRes.data.designers);
        setColorSchemes(colorSchemesRes.data.color_schemes);
      } catch (err) {
        console.error('Failed to load configuration data:', err);
      }
    };
    
    loadConfigData();
    loadHistory();
    loadQueueStatus();
  }, []);

  // Separate useEffect for polling that depends on activeJobs
  useEffect(() => {
    // Set up polling for queue status
    const queueInterval = setInterval(() => {
      console.log('Refreshing queue status...');
      loadQueueStatus();
    }, 2000); // More frequent
    
    // Update active jobs only when there are active jobs
    const jobsInterval = setInterval(() => {
      if (activeJobs.length > 0) {
        console.log('Updating active jobs...');
        updateActiveJobs();
      }
    }, 1500); // More frequent updates
    
    // Refresh history periodically to catch new completions
    const historyInterval = setInterval(() => {
      console.log('Refreshing history...');
      loadHistory();
    }, 3000); // Refresh history every 3 seconds
    
    return () => {
      clearInterval(queueInterval);
      clearInterval(jobsInterval);
      clearInterval(historyInterval);
    };
  }, [activeJobs.length]); // Only re-run when the number of active jobs changes

  const loadHistory = async () => {
    try {
      const response = await axios.get(`${API}/interior-design/history`);
      setJobHistory(response.data.designs || []);
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  };

  const loadQueueStatus = async () => {
    try {
      const response = await axios.get(`${API}/interior-design/queue`);
      setQueueStatus(response.data.queue_status);
    } catch (err) {
      console.error('Failed to load queue status:', err);
    }
  };

  const updateActiveJobs = async () => {
    if (activeJobs.length === 0) return; // No need to update if no active jobs
    
    try {
      console.log(`Updating ${activeJobs.length} active jobs...`);
      
      // Update status for each active job
      const updatedJobs = await Promise.all(
        activeJobs.map(async (job) => {
          if (job.status === 'queued' || job.status === 'processing') {
            try {
              const response = await axios.get(`${API}/interior-design/status/${job.id}`);
              return response.data;
            } catch (err) {
              console.error(`Failed to update job ${job.id}:`, err);
              return job; // Keep original job if update fails
            }
          }
          return job;
        })
      );
      
      // Check for newly completed jobs
      const originallyActive = activeJobs.filter(job => job.status !== 'completed');
      const nowCompleted = updatedJobs.filter(job => job.status === 'completed');
      
      console.log(`Found ${nowCompleted.length} newly completed jobs`);
      
      setActiveJobs(updatedJobs);
      
      // Remove completed jobs from active list and refresh history
      const stillActive = updatedJobs.filter(job => 
        job.status === 'queued' || job.status === 'processing'
      );
      
      if (stillActive.length !== activeJobs.length || nowCompleted.length > 0) {
        console.log('Jobs completed, updating active list and refreshing history');
        setActiveJobs(stillActive);
        // Force refresh history when jobs complete
        loadHistory();
      }
    } catch (err) {
      console.error('Failed to update active jobs:', err);
    }
  };

  const downloadImage = async (designId, filename) => {
    try {
      console.log(`Attempting to download image for design ID: ${designId}`);
      
      // Use fetch to get the image as blob
      const imageUrl = `${API}/images/${designId}.jpg`;
      console.log(`Fetching image from: ${imageUrl}`);
      
      const response = await fetch(imageUrl, {
        method: 'GET',
        credentials: 'same-origin'
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
      }
      
      // Convert to blob
      const blob = await response.blob();
      console.log(`Image blob received, size: ${blob.size} bytes`);
      
      // Create blob URL and download
      const blobUrl = URL.createObjectURL(blob);
      
      // Create download link
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = filename || `ai_design_${designId}.jpg`;
      link.style.display = 'none';
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up blob URL
      URL.revokeObjectURL(blobUrl);
      
      console.log('Image download completed successfully');
      
    } catch (err) {
      console.error('Failed to download image:', err);
      setError(`Failed to download image: ${err.message}`);
      
      // Fallback: open image in new tab
      try {
        const fallbackUrl = `${API}/images/${designId}.jpg`;
        window.open(fallbackUrl, '_blank');
        console.log('Opened image in new tab as fallback');
      } catch (fallbackErr) {
        console.error('Fallback also failed:', fallbackErr);
      }
    }
  };

  const deleteImage = async (designId, filename) => {
    console.log(`Delete button clicked for design: ${designId}, filename: ${filename}`);
    
    if (!window.confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
      console.log('Delete cancelled by user');
      return;
    }
    
    try {
      console.log(`Proceeding with deletion of design: ${designId}`);
      
      const response = await axios.delete(`${API}/interior-design/delete/${designId}`);
      console.log('Delete response:', response.data);
      
      if (response.data.success) {
        console.log('Design deleted successfully from backend');
        
        // Remove from history immediately for better UX
        setJobHistory(prev => {
          const updated = prev.filter(job => job.id !== designId);
          console.log(`Removed from history. New count: ${updated.length}`);
          return updated;
        });
        
        // Refresh history to ensure it's up to date
        setTimeout(() => {
          console.log('Refreshing history after delete...');
          loadHistory();
        }, 1000);
      }
    } catch (err) {
      console.error('Failed to delete design:', err);
      setError(`Failed to delete design: ${err.response?.data?.detail || err.message}`);
    }
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploadedFile(file);
    setError('');
    setProcessedImage(null);
  }, []);

  const handleSubmit = async () => {
    if (!uploadedFile) {
      setError('Please upload an image first');
      return;
    }

    setError('');
    setStatusMessage('Adding to queue...');

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('room_type', selectedRoomType);
    formData.append('designer', selectedDesigner);
    formData.append('color_scheme', selectedColorScheme);

    try {
      const response = await axios.post(`${API}/interior-design/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data) {
        // Add to active jobs queue
        setActiveJobs(prev => [...prev, response.data]);
        setStatusMessage('Added to queue! You can submit more images.');
        
        // Clear the current upload to allow new submissions
        setUploadedFile(null);
        
        // Update queue status
        loadQueueStatus();
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to add to queue');
      setStatusMessage('');
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    maxFiles: 1,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  // Helper function to render toggle buttons
  // Image Viewer Modal Component
  const ImageViewerModal = ({ image, isOpen, onClose }) => {
    if (!isOpen || !image) return null;

    return (
      <div 
        className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4" 
        onClick={onClose}
      >
        <div 
          className="relative max-w-4xl max-h-full bg-white rounded-lg overflow-hidden shadow-2xl" 
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b bg-gray-50">
            <div>
              <h3 className="text-lg font-bold text-gray-900">{image.original_filename}</h3>
              <div className="text-sm text-gray-600">
                {roomTypes.find(r => r.id === image.room_type)?.name} • {designers.find(d => d.id === image.designer)?.name} • {colorSchemes.find(c => c.id === image.color_scheme)?.name}
              </div>
            </div>
            <button 
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 text-2xl font-bold leading-none"
            >
              ×
            </button>
          </div>
          
          {/* Image */}
          <div className="relative">
            <img
              src={image.processed_image_url}
              alt="Generated Design"
              className="w-full h-auto max-h-[70vh] object-contain"
            />
          </div>
          
          {/* Footer with actions */}
          <div className="flex items-center justify-between p-4 border-t bg-gray-50">
            <div className="text-sm text-gray-500">
              Click outside to close or use the × button
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => downloadImage(image.id, image.original_filename)}
                className="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700 transition-colors"
              >
                Download
              </button>
              <button
                onClick={() => {
                  deleteImage(image.id, image.original_filename);
                  onClose();
                }}
                className="bg-red-600 text-white px-4 py-2 rounded text-sm hover:bg-red-700 transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Designer Modal Component
  const DesignerModal = ({ designer, isOpen, onClose }) => {
    if (!isOpen || !designer) return null;

    return (
      <div 
        className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50" 
        onClick={onClose}
      >
        <div 
          className="bg-white rounded-lg shadow-2xl max-w-md w-full mx-4 max-h-96 flex flex-col" 
          onClick={e => e.stopPropagation()}
        >
          {/* Header - Fixed */}
          <div className="flex items-center p-4 border-b border-gray-200 flex-shrink-0">
            <img 
              src={designer.image_url} 
              alt={designer.name}
              className="w-12 h-12 rounded-full object-cover mr-3 border-2 border-gray-200"
            />
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-bold text-gray-900 truncate">{designer.name}</h3>
              <p className="text-sm text-blue-600 font-medium truncate">{designer.description}</p>
            </div>
            <button 
              onClick={onClose}
              className="ml-2 text-gray-400 hover:text-gray-600 text-2xl font-bold leading-none flex-shrink-0"
            >
              ×
            </button>
          </div>
          
          {/* Content - Scrollable */}
          <div className="flex-1 overflow-y-auto p-4">
            <div className="text-gray-700 whitespace-pre-line leading-relaxed text-sm">
              {designer.full_bio}
            </div>
          </div>
          
          {/* Footer - Fixed */}
          <div className="p-4 border-t border-gray-200 flex-shrink-0">
            <button 
              onClick={onClose}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderDesignerGroup = (designers, selected, onSelect, title) => (
    <div className="space-y-3">
      <div>
        <h4 className="text-lg font-semibold text-gray-900">{title}</h4>
        <p className="text-sm text-blue-600 mt-1">Double-click for full bio</p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {designers.map((designer) => (
          <div key={designer.id} className="relative">
            <button
              onClick={(e) => {
                if (e.detail === 1) {
                  // Single click - select designer
                  onSelect(designer.id);
                } else if (e.detail === 2) {
                  // Double click - show modal
                  e.preventDefault();
                  setSelectedDesignerForModal(designer);
                  setShowDesignerModal(true);
                }
              }}
              className={`w-full p-3 rounded-lg border-2 text-left transition-all ${
                selected === designer.id
                  ? 'border-blue-500 bg-blue-50 text-blue-900'
                  : 'border-gray-200 bg-white hover:border-blue-300 text-gray-700'
              }`}
            >
              <div className="flex flex-col items-center">
                <img 
                  src={designer.image_url} 
                  alt={designer.name}
                  className="w-12 h-12 rounded-full object-cover mb-2 border-2 border-gray-200"
                />
                <div className="font-medium text-sm text-center">{designer.name}</div>
                <div className="text-xs mt-1 opacity-75 text-center">{designer.description}</div>
              </div>
            </button>
          </div>
        ))}
      </div>
    </div>
  );

  const renderToggleGroup = (items, selected, onSelect, title) => (
    <div className="space-y-3">
      <h4 className="text-lg font-semibold text-gray-900">{title}</h4>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelect(item.id)}
            className={`p-3 rounded-lg border-2 text-left transition-all ${
              selected === item.id
                ? 'border-blue-500 bg-blue-50 text-blue-900'
                : 'border-gray-200 bg-white hover:border-blue-300 text-gray-700'
            }`}
          >
            <div className="font-medium text-sm">{item.name}</div>
            {item.description && (
              <div className="text-xs mt-1 opacity-75">{item.description}</div>
            )}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <section id="interior-design" className="py-16 bg-white">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">AI Interior Design Enhancement</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload any interior photo and customize your design with our AI-powered styling options.
            Choose your room type, designer aesthetic, and color scheme for personalized results.
          </p>
        </div>

        <div className="max-w-7xl mx-auto">
          {/* Two-column layout */}
          <div className="grid grid-cols-12 gap-6">
            
            {/* Left Column - Queue and History (1/3 width) */}
            <div className="col-span-4 space-y-6">
              
              {/* Queue Status */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-blue-900 mb-2">Queue Status</h3>
                <div className="text-sm space-y-1">
                  <div className="flex justify-between">
                    <span>Queued:</span>
                    <span className="font-medium">{queueStatus.queued}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Processing:</span>
                    <span className="font-medium">{queueStatus.processing}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Active Jobs:</span>
                    <span className="font-medium">{activeJobs.length}</span>
                  </div>
                </div>
              </div>

              {/* Active Jobs */}
              {activeJobs.length > 0 && (
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Active Jobs</h3>
                  <div className="space-y-3 max-h-60 overflow-y-auto">
                    {activeJobs.map((job) => (
                      <div key={job.id} className="bg-gray-50 rounded-lg p-3 border">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-900 truncate">
                            {job.original_filename}
                          </span>
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            job.status === 'queued' ? 'bg-yellow-100 text-yellow-800' :
                            job.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {job.status}
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 space-y-1">
                          <div>Room: {roomTypes.find(r => r.id === job.room_type)?.name}</div>
                          <div>Designer: {designers.find(d => d.id === job.designer)?.name}</div>
                          <div>Colors: {colorSchemes.find(c => c.id === job.color_scheme)?.name}</div>
                        </div>
                        {job.status === 'processing' && (
                          <div className="mt-2">
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* History - Always visible */}
              <div className="bg-white rounded-lg border p-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Recent Designs</h3>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {jobHistory.filter(job => job.status === 'completed').slice(0, 10).map((job) => (
                    <div key={job.id} className="bg-gray-50 rounded-lg overflow-hidden border">
                      {job.processed_image_url && (
                        <img
                          src={job.processed_image_url}
                          alt="Generated Design"
                          className="w-full h-24 object-cover cursor-pointer hover:opacity-90 transition-opacity"
                          onClick={() => setViewingImage(job)}
                          title="Click to view larger"
                        />
                      )}
                      <div className="p-3">
                        <div className="text-sm font-medium text-gray-900 mb-2 truncate">
                          {job.original_filename}
                        </div>
                        <div className="text-xs text-gray-600 space-y-1 mb-2">
                          <div>Room: {roomTypes.find(r => r.id === job.room_type)?.name}</div>
                          <div>Designer: {designers.find(d => d.id === job.designer)?.name}</div>
                          <div>Colors: {colorSchemes.find(c => c.id === job.color_scheme)?.name}</div>
                        </div>
                        <div className="flex space-x-2">
                          <button
                            onClick={() => downloadImage(job.id, job.original_filename)}
                            className="flex-1 bg-blue-600 text-white py-1 px-2 rounded text-xs hover:bg-blue-700 transition-colors"
                          >
                            Download
                          </button>
                          <button
                            onClick={() => deleteImage(job.id, job.original_filename)}
                            className="bg-red-600 text-white py-1 px-2 rounded text-xs hover:bg-red-700 transition-colors"
                            title="Delete design"
                          >
                            🗑️
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right Column - Settings (2/3 width) */}
            <div className="col-span-8 space-y-6">
              
              {/* Step 1: Upload Image */}
              <div className="bg-gray-50 rounded-xl p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Step 1: Upload Your Interior Photo</h3>
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer
                    ${dragActive || isDragActive 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                    }`}
                >
                  <input {...getInputProps()} />
                  <div className="space-y-3">
                    <div className="w-12 h-12 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
                      <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">
                        {uploadedFile ? `Selected: ${uploadedFile.name}` : 'Drop your interior photo here'}
                      </p>
                      <p className="text-gray-500 text-sm">or click to browse</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Step 2: Room Type */}
              <div className="bg-gray-50 rounded-xl p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Step 2: Select Room Type</h3>
                {renderToggleGroup(roomTypes, selectedRoomType, setSelectedRoomType, "")}
              </div>
              
              {/* Step 3: Designer Style - Compact with scroll */}
              <div className="bg-gray-50 rounded-xl p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-2">Step 3: Choose Designer Style</h3>
                <p className="text-sm text-blue-600 mb-4">Double-click for full bio</p>
                <div className="max-h-64 overflow-y-auto border rounded-lg bg-white p-4">
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    {designers.map((designer) => (
                      <div key={designer.id} className="relative">
                        <button
                          onClick={(e) => {
                            if (e.detail === 1) {
                              // Single click - select designer
                              setSelectedDesigner(designer.id);
                            } else if (e.detail === 2) {
                              // Double click - show modal
                              e.preventDefault();
                              setSelectedDesignerForModal(designer);
                              setShowDesignerModal(true);
                            }
                          }}
                          className={`w-full p-3 rounded-lg border-2 text-left transition-all ${
                            selectedDesigner === designer.id
                              ? 'border-blue-500 bg-blue-50 text-blue-900'
                              : 'border-gray-200 bg-white hover:border-blue-300 text-gray-700'
                          }`}
                        >
                          <div className="flex flex-col items-center">
                            <img 
                              src={designer.image_url} 
                              alt={designer.name}
                              className="w-10 h-10 rounded-full object-cover mb-2 border-2 border-gray-200"
                            />
                            <div className="font-medium text-xs text-center">{designer.name}</div>
                            <div className="text-xs mt-1 opacity-75 text-center line-clamp-2">{designer.description}</div>
                          </div>
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Step 4: Color Scheme - Compact with scroll */}
              <div className="bg-gray-50 rounded-xl p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Step 4: Pick Color Scheme</h3>
                <div className="max-h-64 overflow-y-auto border rounded-lg bg-white p-4">
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    {colorSchemes.map((item) => (
                      <button
                        key={item.id}
                        onClick={() => setSelectedColorScheme(item.id)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${
                          selectedColorScheme === item.id
                            ? 'border-blue-500 bg-blue-50 text-blue-900'
                            : 'border-gray-200 bg-white hover:border-blue-300 text-gray-700'
                        }`}
                      >
                        <div className="font-medium text-xs">{item.name}</div>
                        <div className="text-xs mt-1 opacity-75 line-clamp-2">{item.description}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Step 5: Submit Button */}
              <div className="text-center bg-white rounded-xl p-6 border">
                <button
                  onClick={handleSubmit}
                  disabled={!uploadedFile}
                  className={`px-8 py-4 rounded-lg font-semibold text-lg transition-all ${
                    !uploadedFile
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700 transform hover:scale-105'
                  }`}
                >
                  Add to Queue
                </button>
                {!uploadedFile && (
                  <p className="text-gray-500 text-sm mt-2">Please upload an image first</p>
                )}
                {statusMessage && (
                  <p className="text-blue-600 text-sm mt-2">{statusMessage}</p>
                )}
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mt-6">
              <p className="text-red-800">{error}</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Image Viewer Modal */}
      <ImageViewerModal 
        image={viewingImage} 
        isOpen={!!viewingImage} 
        onClose={() => setViewingImage(null)} 
      />
      
      {/* Designer Bio Modal */}
      <DesignerModal 
        designer={selectedDesignerForModal} 
        isOpen={showDesignerModal} 
        onClose={() => {
          setShowDesignerModal(false);
          setSelectedDesignerForModal(null);
        }} 
      />
    </section>
  );
};

// GPT Tools Component
const GPTTools = () => {
  const [activeTab, setActiveTab] = useState('property_description');
  const [formData, setFormData] = useState({});
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const tools = [
    {
      id: 'property_description',
      name: 'Property Description Generator',
      description: 'Generate compelling property descriptions',
      placeholder: 'Enter property details (bedrooms, bathrooms, location, features...)',
      field: 'property_details'
    },
    {
      id: 'market_analysis',
      name: 'Market Analysis',
      description: 'Get comprehensive market insights',
      fields: ['location', 'property_type'],
      placeholders: ['Enter location (e.g., Downtown Seattle)', 'Property type (e.g., Condo, Single-family)']
    },
    {
      id: 'email_template',
      name: 'Email Templates',
      description: 'Professional client communication',
      fields: ['email_type', 'context'],
      placeholders: ['Email type (e.g., Follow-up, Listing inquiry)', 'Context/situation']
    }
  ];

  const handleSubmit = async (toolId) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/gpt-concepts/${toolId.replace('_', '-')}`, formData);
      setResponse(response.data);
    } catch (err) {
      setResponse({ error: err.response?.data?.detail || 'Failed to generate response' });
    } finally {
      setLoading(false);
    }
  };

  const activeTool = tools.find(tool => tool.id === activeTab);

  return (
    <section id="ai-tools" className="py-16 bg-gray-50">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">Professional AI Tools</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Access 30+ specialized AI tools designed for real estate professionals. 
            Generate content, analyze markets, and streamline your workflow.
          </p>
        </div>

        <div className="max-w-6xl mx-auto">
          <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
            {/* Tool Tabs */}
            <div className="border-b border-gray-200">
              <div className="flex flex-wrap">
                {tools.map((tool) => (
                  <button
                    key={tool.id}
                    onClick={() => {
                      setActiveTab(tool.id);
                      setResponse(null);
                      setFormData({});
                    }}
                    className={`px-6 py-4 font-medium text-sm transition-colors ${
                      activeTab === tool.id
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-600 hover:text-blue-600'
                    }`}
                  >
                    {tool.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Tool Content */}
            <div className="p-8">
              <div className="grid lg:grid-cols-2 gap-8">
                {/* Input Section */}
                <div className="space-y-6">
                  <div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">{activeTool.name}</h3>
                    <p className="text-gray-600">{activeTool.description}</p>
                  </div>

                  <div className="space-y-4">
                    {activeTool.field ? (
                      <textarea
                        placeholder={activeTool.placeholder}
                        value={formData[activeTool.field] || ''}
                        onChange={(e) => setFormData({ ...formData, [activeTool.field]: e.target.value })}
                        className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        rows="4"
                      />
                    ) : (
                      activeTool.fields.map((field, index) => (
                        <input
                          key={field}
                          type="text"
                          placeholder={activeTool.placeholders[index]}
                          value={formData[field] || ''}
                          onChange={(e) => setFormData({ ...formData, [field]: e.target.value })}
                          className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                      ))
                    )}

                    <button
                      onClick={() => handleSubmit(activeTool.id)}
                      disabled={loading}
                      className="w-full bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
                    >
                      {loading ? 'Generating...' : 'Generate with AI'}
                    </button>
                  </div>
                </div>

                {/* Output Section */}
                <div className="bg-gray-50 rounded-lg p-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-4">Generated Content</h4>
                  {response ? (
                    response.error ? (
                      <div className="bg-red-50 border border-red-200 rounded p-4 text-red-800">
                        {response.error}
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="bg-white rounded p-4 border">
                          <pre className="whitespace-pre-wrap text-gray-900 text-sm">{response.response}</pre>
                        </div>
                        <div className="text-xs text-gray-500">
                          Generated: {new Date(response.timestamp).toLocaleString()}
                        </div>
                      </div>
                    )
                  ) : (
                    <div className="text-gray-500 text-center py-8">
                      Fill in the form and click "Generate with AI" to see results
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

// Features Section Component
const FeaturesSection = () => {
  const features = [
    {
      title: "Interior Design AI",
      description: "Transform any space with our custom-trained model that generates professional interior designs instantly.",
      image: "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2Njd8MHwxfHNlYXJjaHwyfHx0ZWNobm9sb2d5fGVufDB8fHxibHVlfDE3NTMwMjcxODZ8MA&ixlib=rb-4.1.0&q=85",
      icon: "🏠"
    },
    {
      title: "AI-Powered Tools",
      description: "Access 30+ specialized tools for property descriptions, market analysis, client communications and more.",
      image: "https://images.unsplash.com/photo-1581090464777-f3220bbe1b8b?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2Njd8MHwxfHNlYXJjaHwzfHx0ZWNobm9sb2d5fGVufDB8fHxibHVlfDE3NTMwMjcxODZ8MA&ixlib=rb-4.1.0&q=85",
      icon: "🤖"
    },
    {
      title: "Credit-Based Pricing",
      description: "Flexible tiered pricing system that scales with your business needs and usage patterns.",
      icon: "💳"
    }
  ];

  return (
    <section className="py-16 bg-white">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">Why Choose ProAgentTools?</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Built specifically for real estate professionals, our AI tools help you close more deals, 
            impress clients, and grow your business.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="text-center">
              {feature.image ? (
                <div className="mb-6">
                  <img 
                    src={feature.image}
                    alt={feature.title}
                    className="w-full h-48 object-cover rounded-lg shadow-lg"
                  />
                </div>
              ) : (
                <div className="w-16 h-16 mx-auto mb-6 bg-blue-100 rounded-full flex items-center justify-center">
                  <span className="text-3xl">{feature.icon}</span>
                </div>
              )}
              <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Pricing Section Component
const PricingSection = () => (
  <section id="pricing" className="py-16 bg-gray-50">
    <div className="container mx-auto px-6">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Simple, Fair Pricing</h2>
        <p className="text-xl text-gray-600">Pay for what you use with our flexible credit system</p>
      </div>

      <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
        {[
          { name: "Starter", price: "$29", credits: "100 Credits", popular: false },
          { name: "Professional", price: "$79", credits: "300 Credits", popular: true },
          { name: "Enterprise", price: "$199", credits: "1000 Credits", popular: false }
        ].map((plan, index) => (
          <div key={index} className={`bg-white rounded-2xl shadow-lg p-8 ${plan.popular ? 'border-2 border-blue-500' : ''}`}>
            {plan.popular && (
              <div className="bg-blue-500 text-white text-sm font-medium px-3 py-1 rounded-full inline-block mb-4">
                Most Popular
              </div>
            )}
            <h3 className="text-2xl font-bold text-gray-900 mb-2">{plan.name}</h3>
            <div className="mb-6">
              <span className="text-4xl font-bold text-gray-900">{plan.price}</span>
              <span className="text-gray-600">/month</span>
            </div>
            <p className="text-gray-600 mb-6">{plan.credits} per month</p>
            <button className={`w-full py-3 px-6 rounded-lg font-semibold transition-colors ${
              plan.popular 
                ? 'bg-blue-600 text-white hover:bg-blue-700' 
                : 'bg-gray-200 text-gray-900 hover:bg-gray-300'
            }`}>
              Get Started
            </button>
          </div>
        ))}
      </div>
    </div>
  </section>
);

// Footer Component
const Footer = () => (
  <footer className="bg-gray-900 text-white py-12">
    <div className="container mx-auto px-6">
      <div className="grid md:grid-cols-4 gap-8">
        <div>
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-blue-400 rounded-lg flex items-center justify-center">
              <span className="text-lg font-bold">PA</span>
            </div>
            <span className="text-xl font-bold">ProAgentTools</span>
          </div>
          <p className="text-gray-400">AI-powered tools for real estate professionals</p>
        </div>
        <div>
          <h4 className="text-lg font-semibold mb-4">Product</h4>
          <ul className="space-y-2 text-gray-400">
            <li><a href="#" className="hover:text-white transition-colors">Interior Design AI</a></li>
            <li><a href="#" className="hover:text-white transition-colors">GPT Tools</a></li>
            <li><a href="#" className="hover:text-white transition-colors">API Access</a></li>
          </ul>
        </div>
        <div>
          <h4 className="text-lg font-semibold mb-4">Company</h4>
          <ul className="space-y-2 text-gray-400">
            <li><a href="#" className="hover:text-white transition-colors">About</a></li>
            <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
            <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
          </ul>
        </div>
        <div>
          <h4 className="text-lg font-semibold mb-4">Support</h4>
          <ul className="space-y-2 text-gray-400">
            <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
            <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
            <li><a href="#" className="hover:text-white transition-colors">Referral Program</a></li>
          </ul>
        </div>
      </div>
      <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
        <p>&copy; 2025 ProAgentTools. All rights reserved.</p>
      </div>
    </div>
  </footer>
);

// Main App Component
function App() {
  return (
    <div className="App">
      <Header />
      <HeroSection />
      <InteriorDesignTool />
      <GPTTools />
      <FeaturesSection />
      <PricingSection />
      <Footer />
    </div>
  );
}

export default App;