import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import axios from 'axios';

const CreateListingPage = ({ onClose, onListingCreated }) => {
  const { user, token } = useAuth();
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  
  const [formData, setFormData] = useState({
    // Property Details
    address: '',
    city: '',
    state: '',
    zip_code: '',
    beds: 1,
    baths: 1,
    sqft: '',
    lot_size_sqft: '',
    year_built: '',
    property_type: 'Single Family',
    listing_price: '',
    mls_number: '',
    // Listing Details
    description: '',
    agent_notes: ''
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const propertyTypes = [
    'Single Family', 'Condo', 'Townhouse', 'Multi-Family', 
    'Land', 'Commercial', 'Mobile Home'
  ];

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? (value === '' ? '' : Number(value)) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Check if user has enough credits
    if (user.credits < 20) {
      setError('You need at least 20 credits to create a listing with foundation content.');
      return;
    }

    await createListing();
  };

  const createListing = async () => {
    setLoading(true);

    try {
      const listingData = {
        property_details: {
          address: formData.address,
          city: formData.city,
          state: formData.state,
          zip_code: formData.zip_code,
          beds: formData.beds,
          baths: formData.baths,
          sqft: formData.sqft || null,
          lot_size_sqft: formData.lot_size_sqft || null,
          year_built: formData.year_built || null,
          property_type: formData.property_type,
          listing_price: formData.listing_price || null,
          mls_number: formData.mls_number || null,
        },
        description: formData.description || null,
        selected_tool_ids: [],
        agent_notes: formData.agent_notes || null
      };

      const response = await axios.post(`${BACKEND_URL}/api/listings`, listingData, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (onListingCreated) {
        onListingCreated(response.data);
      }

    } catch (err) {
      console.error('Create listing error:', err);
      setError(err.response?.data?.detail || 'Failed to create listing');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-6">
        
        {/* Header */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Create New Listing</h1>
              <p className="text-gray-600 mt-1">Add a property and select AI tools to enhance your listing</p>
            </div>
            {onClose && (
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600"
              >
                <span className="text-2xl">×</span>
              </button>
            )}
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          
          {/* Property Information */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Property Information</h2>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Street Address *
                </label>
                <input
                  type="text"
                  name="address"
                  value={formData.address}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="123 Main Street"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">City *</label>
                <input
                  type="text"
                  name="city"
                  value={formData.city}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">State *</label>
                <input
                  type="text"
                  name="state"
                  value={formData.state}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="CA"
                  maxLength="2"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">ZIP Code *</label>
                <input
                  type="text"
                  name="zip_code"
                  value={formData.zip_code}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="90210"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Property Type *</label>
                <select
                  name="property_type"
                  value={formData.property_type}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  {propertyTypes.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid md:grid-cols-4 gap-4 mt-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Beds *</label>
                <input
                  type="number"
                  name="beds"
                  value={formData.beds}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Baths *</label>
                <input
                  type="number"
                  name="baths"
                  value={formData.baths}
                  onChange={handleInputChange}
                  min="0"
                  step="0.5"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Sq Ft</label>
                <input
                  type="number"
                  name="sqft"
                  value={formData.sqft}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="1,500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Year Built</label>
                <input
                  type="number"
                  name="year_built"
                  value={formData.year_built}
                  onChange={handleInputChange}
                  min="1800"
                  max={new Date().getFullYear()}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="2020"
                />
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4 mt-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Listing Price</label>
                <input
                  type="number"
                  name="listing_price"
                  value={formData.listing_price}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="750000"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">MLS Number</label>
                <input
                  type="text"
                  name="mls_number"
                  value={formData.mls_number}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="ML12345678"
                />
              </div>
            </div>
          </div>

          {/* Property Description */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Description & Notes</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Property Description</label>
                <textarea
                  name="description"
                  value={formData.description}
                  onChange={handleInputChange}
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter initial property description or key features..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Agent Notes</label>
                <textarea
                  name="agent_notes"
                  value={formData.agent_notes}
                  onChange={handleInputChange}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Private notes about the property, client, or strategy..."
                />
              </div>
            </div>
          </div>

          {/* AI Tools Selection */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900">Select AI Tools</h2>
              <div className="text-lg font-medium text-blue-600">
                Total: {calculateTotalCredits()} credits
              </div>
            </div>
            
            {Object.entries(aiTools).map(([category, tools]) => (
              <div key={category} className="mb-6">
                <h3 className="text-lg font-medium text-gray-800 mb-3 border-b pb-2">
                  {category} ({tools.length} tools)
                </h3>
                
                <div className="grid md:grid-cols-2 gap-3">
                  {tools.map(tool => (
                    <div key={tool.id} className="border border-gray-200 rounded-lg p-3 hover:bg-gray-50">
                      <label className="flex items-start space-x-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formData.selected_tool_ids.includes(tool.id)}
                          onChange={(e) => handleToolSelection(tool.id, e.target.checked)}
                          className="mt-1 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                        />
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <h4 className="text-sm font-medium text-gray-900">{tool.name}</h4>
                            <span className="text-sm font-medium text-blue-600">{tool.credits_cost} credits</span>
                          </div>
                          <p className="text-xs text-gray-600 mt-1">{tool.description}</p>
                        </div>
                      </label>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800">{error}</p>
            </div>
          )}

          {/* Submit Button */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">
                  {formData.selected_tool_ids.length} tools selected • {calculateTotalCredits()} credits required
                </p>
              </div>
              <button
                type="submit"
                disabled={loading}
                className={`px-8 py-3 rounded-lg font-semibold transition-colors ${
                  loading 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : 'bg-blue-600 hover:bg-blue-700'
                } text-white`}
              >
                {loading ? 'Creating...' : 'Create Listing'}
              </button>
            </div>
          </div>

        </form>
      </div>

      {/* Credit Confirmation Modal */}
      <CreditConfirmationModal
        isOpen={showCreditConfirmation}
        onConfirm={handleCreditConfirmation}
        onCancel={handleCreditCancel}
        totalCredits={calculateTotalCredits()}
        selectedTools={getSelectedToolsForModal()}
        actionType="create"
        propertyAddress={`${formData.address}, ${formData.city}, ${formData.state}`}
      />
    </div>
  );
};

export default CreateListingPage;