import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import CreateListingPage from './CreateListingPage';
import AIResultsModal from './AIResultsModal';
import CreditConfirmationModal from '../common/CreditConfirmationModal';
import axios from 'axios';

const ListingsDashboard = () => {
  const { user, token } = useAuth();
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  
  const [listings, setListings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedListing, setSelectedListing] = useState(null);
  const [showAIResults, setShowAIResults] = useState(false);
  const [selectedListingForResults, setSelectedListingForResults] = useState(null);

  useEffect(() => {
    loadListings();
  }, []);

  const loadListings = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/listings`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setListings(response.data);
    } catch (err) {
      console.error('Failed to load listings:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleListingCreated = (newListing) => {
    setListings(prev => [newListing, ...prev]);
    setShowCreateForm(false);
  };

  const handleDeleteListing = async (listingId) => {
    if (!window.confirm('Are you sure you want to delete this listing?')) {
      return;
    }

    try {
      await axios.delete(`${BACKEND_URL}/api/listings/${listingId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setListings(prev => prev.filter(l => l.id !== listingId));
    } catch (err) {
      console.error('Failed to delete listing:', err);
      alert('Failed to delete listing. Please try again.');
    }
  };

  const handleProcessAI = async (listingId) => {
    const listing = listings.find(l => l.id === listingId);
    const totalCredits = listing.selected_ai_tools.reduce((sum, tool) => sum + tool.credits_cost, 0);
    
    if (!window.confirm(`Process ${listing.selected_ai_tools.length} AI tools for ${totalCredits} credits?`)) {
      return;
    }

    try {
      // Update UI to show processing
      setListings(prev => prev.map(l => 
        l.id === listingId 
          ? { ...l, ai_processing_status: 'processing' }
          : l
      ));

      const response = await axios.post(`${BACKEND_URL}/api/listings/${listingId}/process-ai`, {}, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.data.success) {
        // Refresh listings to get updated status
        await loadListings();
        alert(`AI processing completed! Used ${response.data.credits_used} credits.`);
      }
    } catch (err) {
      console.error('Failed to process AI:', err);
      alert(err.response?.data?.detail || 'Failed to process AI tools. Please try again.');
      // Revert UI state
      setListings(prev => prev.map(l => 
        l.id === listingId 
          ? { ...l, ai_processing_status: 'pending' }
          : l
      ));
    }
  };

  const handleViewAIResults = (listingId) => {
    setSelectedListingForResults(listingId);
    setShowAIResults(true);
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      draft: { bg: 'bg-gray-100', text: 'text-gray-800', label: 'Draft' },
      active: { bg: 'bg-green-100', text: 'text-green-800', label: 'Active' },
      pending: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Pending' },
      sold: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'Sold' }
    };
    
    const config = statusConfig[status] || statusConfig.draft;
    
    return (
      <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${config.bg} ${config.text}`}>
        {config.label}
      </span>
    );
  };

  const getAIProcessingBadge = (status) => {
    const statusConfig = {
      pending: { bg: 'bg-orange-100', text: 'text-orange-800', label: 'AI Pending', icon: '⏳' },
      processing: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'AI Processing', icon: '🔄' },
      completed: { bg: 'bg-green-100', text: 'text-green-800', label: 'AI Complete', icon: '✅' },
      failed: { bg: 'bg-red-100', text: 'text-red-800', label: 'AI Failed', icon: '❌' }
    };
    
    const config = statusConfig[status] || statusConfig.pending;
    
    return (
      <span className={`inline-flex items-center space-x-1 px-2 py-1 text-xs font-medium rounded-full ${config.bg} ${config.text}`}>
        <span>{config.icon}</span>
        <span>{config.label}</span>
      </span>
    );
  };

  const formatPrice = (price) => {
    if (!price) return 'Price TBD';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(price);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  if (showCreateForm) {
    return (
      <CreateListingPage 
        onClose={() => setShowCreateForm(false)}
        onListingCreated={handleListingCreated}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">My Listings</h1>
              <p className="text-gray-600 mt-1">
                Manage your properties and AI-generated marketing materials
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm">
                <div className="text-gray-600">Available Credits</div>
                <div className="text-2xl font-bold text-blue-600">{user?.credits || 0}</div>
              </div>
              <button
                onClick={() => setShowCreateForm(true)}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                + New Listing
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : listings.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-4xl">🏠</span>
            </div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">No listings yet</h2>
            <p className="text-gray-600 mb-6">
              Create your first listing to start using AI-powered real estate tools
            </p>
            <button
              onClick={() => setShowCreateForm(true)}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Create First Listing
            </button>
          </div>
        ) : (
          <div className="grid gap-6">
            
            {/* Stats Overview */}
            <div className="grid md:grid-cols-4 gap-4 mb-6">
              <div className="bg-white rounded-lg shadow p-4">
                <div className="text-2xl font-bold text-gray-900">{listings.length}</div>
                <div className="text-sm text-gray-600">Total Listings</div>
              </div>
              <div className="bg-white rounded-lg shadow p-4">
                <div className="text-2xl font-bold text-green-600">
                  {listings.filter(l => l.status === 'active').length}
                </div>
                <div className="text-sm text-gray-600">Active</div>
              </div>
              <div className="bg-white rounded-lg shadow p-4">
                <div className="text-2xl font-bold text-blue-600">
                  {listings.filter(l => l.ai_processing_status === 'completed').length}
                </div>
                <div className="text-sm text-gray-600">AI Complete</div>
              </div>
              <div className="bg-white rounded-lg shadow p-4">
                <div className="text-2xl font-bold text-orange-600">
                  {listings.filter(l => l.ai_processing_status === 'pending').length}
                </div>
                <div className="text-sm text-gray-600">AI Pending</div>
              </div>
            </div>

            {/* Listings Grid */}
            <div className="grid lg:grid-cols-2 gap-6">
              {listings.map(listing => (
                <div key={listing.id} className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow">
                  
                  {/* Header */}
                  <div className="p-6 border-b border-gray-200">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-gray-900 mb-1">
                          {listing.property_details.address}
                        </h3>
                        <p className="text-gray-600">
                          {listing.property_details.city}, {listing.property_details.state} {listing.property_details.zip_code}
                        </p>
                        <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
                          <span>{listing.property_details.beds} bed</span>
                          <span>{listing.property_details.baths} bath</span>
                          {listing.property_details.sqft && (
                            <span>{listing.property_details.sqft.toLocaleString()} sq ft</span>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xl font-bold text-gray-900 mb-2">
                          {formatPrice(listing.property_details.listing_price)}
                        </div>
                        <div className="flex items-center space-x-2">
                          {getStatusBadge(listing.status)}
                          {getAIProcessingBadge(listing.ai_processing_status)}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="p-6">
                    
                    {/* AI Tools Summary */}
                    {listing.selected_ai_tools.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-sm font-medium text-gray-900 mb-2">AI Tools Selected</h4>
                        <div className="flex flex-wrap gap-2">
                          {listing.selected_ai_tools.slice(0, 3).map(tool => (
                            <span 
                              key={tool.tool_id}
                              className="inline-flex items-center px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded"
                            >
                              {tool.tool_name}
                              {tool.completed && <span className="ml-1 text-green-600">✓</span>}
                            </span>
                          ))}
                          {listing.selected_ai_tools.length > 3 && (
                            <span className="inline-flex items-center px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded">
                              +{listing.selected_ai_tools.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Description */}
                    {listing.description && (
                      <div className="mb-4">
                        <p className="text-sm text-gray-700 line-clamp-2">{listing.description}</p>
                      </div>
                    )}

                    {/* Meta Info */}
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>Created {formatDate(listing.created_at)}</span>
                      <span>Updated {formatDate(listing.updated_at)}</span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">
                        View Details
                      </button>
                      <button 
                        onClick={() => window.location.href = `#interior-design`}
                        className="text-sm text-green-600 hover:text-green-700 font-medium"
                      >
                        Interior Design
                      </button>
                      {listing.ai_processing_status === 'pending' && listing.selected_ai_tools.length > 0 && (
                        <button 
                          onClick={() => handleProcessAI(listing.id)}
                          className="text-sm text-orange-600 hover:text-orange-700 font-medium"
                        >
                          Process AI ({listing.selected_ai_tools.reduce((sum, tool) => sum + tool.credits_cost, 0)} credits)
                        </button>
                      )}
                      {listing.ai_processing_status === 'completed' && (
                        <button 
                          onClick={() => handleViewAIResults(listing.id)}
                          className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                        >
                          View AI Results
                        </button>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <button className="text-sm text-gray-600 hover:text-gray-700">
                        Edit
                      </button>
                      <button 
                        onClick={() => handleDeleteListing(listing.id)}
                        className="text-sm text-red-600 hover:text-red-700"
                      >
                        Delete
                      </button>
                    </div>
                  </div>

                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* AI Results Modal */}
      {showAIResults && selectedListingForResults && (
        <AIResultsModal 
          listingId={selectedListingForResults}
          onClose={() => {
            setShowAIResults(false);
            setSelectedListingForResults(null);
          }}
        />
      )}
    </div>
  );
};

export default ListingsDashboard;