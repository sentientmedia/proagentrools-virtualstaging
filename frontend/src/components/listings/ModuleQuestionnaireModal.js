import React, { useState } from 'react';

const ModuleQuestionnaireModal = ({ 
  isOpen, 
  onClose, 
  moduleName, 
  writingStyles,
  onSubmit,
  isGenerating
}) => {
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

  if (!isOpen) return null;

  const handleCheckboxChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: prev[field].includes(value)
        ? prev[field].filter(v => v !== value)
        : [...prev[field], value]
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Only send non-empty fields
    const filteredData = {};
    Object.keys(formData).forEach(key => {
      if (formData[key] && (Array.isArray(formData[key]) ? formData[key].length > 0 : formData[key] !== '')) {
        filteredData[key] = formData[key];
      }
    });
    onSubmit(filteredData);
  };

  const handleSkip = () => {
    onSubmit({});
  };

  // Define module-specific questions
  const getQuestionsForModule = () => {
    const moduleQuestions = {
      marketing_copy: {
        title: 'Marketing Materials Context',
        questions: [
          {
            field: 'tone',
            label: 'Writing Style/Tone',
            type: 'select',
            required: true,
            options: writingStyles.map(style => ({ value: style.id, label: `${style.name} - ${style.description}` }))
          },
          {
            field: 'target_buyer_type',
            label: 'Target Buyer Type',
            type: 'select',
            required: true,
            options: [
              { value: 'first_time_homebuyer', label: 'First-Time Homebuyer' },
              { value: 'young_families', label: 'Young Families' },
              { value: 'retirees', label: 'Retirees/Downsizers' },
              { value: 'professionals', label: 'Young Professionals' },
              { value: 'investors', label: 'Investors' },
              { value: 'luxury_buyers', label: 'Luxury Buyers' },
              { value: 'relocating', label: 'Relocating Families' }
            ]
          },
          {
            field: 'property_highlights',
            label: 'Key Selling Points (Select all that apply)',
            type: 'checkbox',
            options: [
              { value: 'updated_kitchen', label: 'Updated Kitchen' },
              { value: 'modern_bathrooms', label: 'Modern Bathrooms' },
              { value: 'large_yard', label: 'Large Yard/Outdoor Space' },
              { value: 'prime_location', label: 'Prime Location' },
              { value: 'top_schools', label: 'Top-Rated Schools' },
              { value: 'walkability', label: 'Walkable Neighborhood' },
              { value: 'recent_renovations', label: 'Recent Renovations' },
              { value: 'energy_efficient', label: 'Energy Efficient Features' },
              { value: 'smart_home', label: 'Smart Home Technology' },
              { value: 'garage_parking', label: 'Garage/Parking' }
            ]
          },
          {
            field: 'competitive_advantages',
            label: 'Competitive Advantages (Select all that apply)',
            type: 'checkbox',
            options: [
              { value: 'priced_competitively', label: 'Priced Below Comparables' },
              { value: 'move_in_ready', label: 'Move-In Ready' },
              { value: 'unique_features', label: 'Unique Features' },
              { value: 'low_hoa', label: 'Low/No HOA Fees' },
              { value: 'larger_lot', label: 'Larger Lot Size' },
              { value: 'better_condition', label: 'Better Condition Than Comps' }
            ]
          }
        ]
      },
      social_media: {
        title: 'Social Media Posts Context',
        questions: [
          {
            field: 'tone',
            label: 'Writing Style/Tone',
            type: 'select',
            required: true,
            options: writingStyles.map(style => ({ value: style.id, label: `${style.name} - ${style.description}` }))
          },
          {
            field: 'target_buyer_type',
            label: 'Target Audience',
            type: 'select',
            options: [
              { value: 'first_time_homebuyer', label: 'First-Time Homebuyer' },
              { value: 'young_families', label: 'Young Families' },
              { value: 'retirees', label: 'Retirees' },
              { value: 'professionals', label: 'Young Professionals' },
              { value: 'investors', label: 'Investors' },
              { value: 'luxury_buyers', label: 'Luxury Buyers' }
            ]
          },
          {
            field: 'property_highlights',
            label: 'Features to Emphasize',
            type: 'checkbox',
            options: [
              { value: 'updated_kitchen', label: 'Updated Kitchen' },
              { value: 'outdoor_space', label: 'Outdoor Space' },
              { value: 'location', label: 'Great Location' },
              { value: 'schools', label: 'Top-Rated Schools' },
              { value: 'modern_design', label: 'Modern Design' }
            ]
          }
        ]
      },
      open_house_promo: {
        title: 'Open House Details',
        questions: [
          {
            field: 'tone',
            label: 'Writing Style/Tone',
            type: 'select',
            required: true,
            options: writingStyles.map(style => ({ value: style.id, label: `${style.name} - ${style.description}` }))
          },
          {
            field: 'open_house_date',
            label: 'Open House Date',
            type: 'date',
            required: true
          },
          {
            field: 'open_house_time',
            label: 'Open House Time',
            type: 'select',
            required: true,
            options: [
              { value: '10am_12pm', label: '10:00 AM - 12:00 PM' },
              { value: '12pm_2pm', label: '12:00 PM - 2:00 PM' },
              { value: '1pm_3pm', label: '1:00 PM - 3:00 PM' },
              { value: '2pm_4pm', label: '2:00 PM - 4:00 PM' }
            ]
          },
          {
            field: 'open_house_features',
            label: 'Special Features (Select all that apply)',
            type: 'checkbox',
            options: [
              { value: 'refreshments', label: 'Refreshments Provided' },
              { value: 'giveaways', label: 'Door Prizes/Giveaways' },
              { value: 'guided_tours', label: 'Guided Tours Available' },
              { value: 'rsvp_required', label: 'RSVP Required' },
              { value: 'parking_available', label: 'Ample Parking' }
            ]
          }
        ]
      },
      virtual_tour_script: {
        title: 'Virtual Tour Script Context',
        questions: [
          {
            field: 'tone',
            label: 'Writing Style/Tone',
            type: 'select',
            required: true,
            options: writingStyles.map(style => ({ value: style.id, label: `${style.name} - ${style.description}` }))
          },
          {
            field: 'video_length',
            label: 'Target Video Length',
            type: 'select',
            required: true,
            options: [
              { value: '30_seconds', label: '30 seconds (Quick Overview)' },
              { value: '1_minute', label: '1 minute (Standard)' },
              { value: '2_minutes', label: '2 minutes (Detailed)' },
              { value: '3_plus_minutes', label: '3+ minutes (Comprehensive)' }
            ]
          },
          {
            field: 'rooms_to_highlight',
            label: 'Rooms to Highlight (Select priority areas)',
            type: 'checkbox',
            options: [
              { value: 'entrance_foyer', label: 'Entrance/Foyer' },
              { value: 'living_room', label: 'Living Room' },
              { value: 'kitchen', label: 'Kitchen' },
              { value: 'dining_room', label: 'Dining Room' },
              { value: 'master_bedroom', label: 'Master Bedroom' },
              { value: 'bathrooms', label: 'Bathrooms' },
              { value: 'outdoor_space', label: 'Outdoor Space/Yard' },
              { value: 'special_features', label: 'Special Features' }
            ]
          }
        ]
      },
      buyer_profile: {
        title: 'Target Buyer Profile Context',
        questions: [
          {
            field: 'tone',
            label: 'Writing Style/Tone',
            type: 'select',
            required: true,
            options: writingStyles.map(style => ({ value: style.id, label: `${style.name} - ${style.description}` }))
          },
          {
            field: 'target_buyer_type',
            label: 'Primary Target Buyer',
            type: 'select',
            required: true,
            options: [
              { value: 'first_time_homebuyer', label: 'First-Time Homebuyer' },
              { value: 'young_families', label: 'Young Families (Kids at Home)' },
              { value: 'retirees', label: 'Retirees/Empty Nesters' },
              { value: 'professionals', label: 'Young Professionals/DINKS' },
              { value: 'investors', label: 'Real Estate Investors' },
              { value: 'luxury_buyers', label: 'Luxury Buyers' },
              { value: 'relocating', label: 'Relocating Families' }
            ]
          }
        ]
      },
      price_justification: {
        title: 'Price Justification Context',
        questions: [
          {
            field: 'tone',
            label: 'Writing Style/Tone',
            type: 'select',
            required: true,
            options: writingStyles.map(style => ({ value: style.id, label: `${style.name} - ${style.description}` }))
          },
          {
            field: 'pricing_strategy',
            label: 'Pricing Strategy',
            type: 'select',
            required: true,
            options: [
              { value: 'competitive', label: 'Competitively Priced (At Market Value)' },
              { value: 'premium', label: 'Premium Pricing (Above Comps)' },
              { value: 'aggressive', label: 'Aggressive Pricing (Below Market)' },
              { value: 'negotiable', label: 'Price Negotiable' }
            ]
          },
          {
            field: 'recent_upgrades',
            label: 'Recent Upgrades/Improvements',
            type: 'checkbox',
            options: [
              { value: 'new_roof', label: 'New Roof' },
              { value: 'hvac_system', label: 'New HVAC System' },
              { value: 'kitchen_remodel', label: 'Kitchen Remodel' },
              { value: 'bathroom_updates', label: 'Bathroom Updates' },
              { value: 'flooring', label: 'New Flooring' },
              { value: 'windows', label: 'New Windows' },
              { value: 'landscaping', label: 'Professional Landscaping' },
              { value: 'smart_features', label: 'Smart Home Features' }
            ]
          }
        ]
      },
      objection_handling: {
        title: 'Objection Handling Context',
        questions: [
          {
            field: 'tone',
            label: 'Writing Style/Tone',
            type: 'select',
            required: true,
            options: writingStyles.map(style => ({ value: style.id, label: `${style.name} - ${style.description}` }))
          },
          {
            field: 'known_objections',
            label: 'Known Buyer Concerns',
            type: 'checkbox',
            options: [
              { value: 'price_too_high', label: 'Price Concerns' },
              { value: 'needs_updates', label: 'Needs Updates/Repairs' },
              { value: 'small_yard', label: 'Limited Outdoor Space' },
              { value: 'location', label: 'Location Concerns' },
              { value: 'hoa_fees', label: 'HOA Fees' },
              { value: 'older_home', label: 'Age of Home' },
              { value: 'layout', label: 'Floor Plan/Layout' },
              { value: 'parking', label: 'Parking Limitations' }
            ]
          },
          {
            field: 'showing_feedback',
            label: 'Common Showing Feedback',
            type: 'checkbox',
            options: [
              { value: 'too_small', label: 'Size/Space Concerns' },
              { value: 'dark_interior', label: 'Wants More Natural Light' },
              { value: 'busy_street', label: 'Street Noise/Traffic' },
              { value: 'outdated_style', label: 'Outdated Style/Finishes' }
            ]
          }
        ]
      }
    };

    return moduleQuestions[moduleName] || {
      title: 'Module Context',
      questions: [
        {
          field: 'tone',
          label: 'Writing Style/Tone',
          type: 'select',
          required: true,
          options: writingStyles.map(style => ({ value: style.id, label: `${style.name} - ${style.description}` }))
        }
      ]
    };
  };

  const moduleConfig = getQuestionsForModule();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-900">{moduleConfig.title}</h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Provide context to help generate better, more targeted content. All fields are optional.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {moduleConfig.questions.map((question, index) => (
            <div key={index} className="space-y-2">
              <label className="block text-sm font-semibold text-gray-900">
                {question.label}
                {question.required && <span className="text-red-500 ml-1">*</span>}
              </label>

              {question.type === 'select' && (
                <select
                  value={formData[question.field]}
                  onChange={(e) => setFormData(prev => ({ ...prev, [question.field]: e.target.value }))}
                  required={question.required}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">-- Select an option --</option>
                  {question.options.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              )}

              {question.type === 'date' && (
                <input
                  type="date"
                  value={formData[question.field]}
                  onChange={(e) => setFormData(prev => ({ ...prev, [question.field]: e.target.value }))}
                  required={question.required}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              )}

              {question.type === 'checkbox' && (
                <div className="space-y-2 max-h-48 overflow-y-auto border border-gray-200 rounded-lg p-3">
                  {question.options.map(option => (
                    <label key={option.value} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 p-2 rounded">
                      <input
                        type="checkbox"
                        checked={formData[question.field].includes(option.value)}
                        onChange={() => handleCheckboxChange(question.field, option.value)}
                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">{option.label}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          ))}

          {/* Additional Context Field - Available for all modules */}
          <div className="space-y-2 pt-4 border-t border-gray-200">
            <label className="block text-sm font-semibold text-gray-900">
              Additional Notes or Context
              <span className="text-gray-500 font-normal ml-2">(Optional)</span>
            </label>
            <textarea
              value={formData.additional_context}
              onChange={(e) => setFormData(prev => ({ ...prev, additional_context: e.target.value }))}
              placeholder="Add any specific details, preferences, or context that should be included..."
              rows={3}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            />
            <p className="text-xs text-gray-500">
              This information will be used to customize the generated content to your specific needs.
            </p>
          </div>

          <div className="flex items-center justify-between pt-6 border-t border-gray-200">
            <button
              type="button"
              onClick={handleSkip}
              disabled={isGenerating}
              className="px-4 py-2 text-gray-600 hover:text-gray-800 font-medium disabled:text-gray-400"
            >
              Skip & Generate
            </button>
            <div className="flex items-center space-x-3">
              <button
                type="button"
                onClick={onClose}
                disabled={isGenerating}
                className="px-4 py-2 text-gray-600 hover:text-gray-800 font-medium disabled:text-gray-400"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isGenerating}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors disabled:bg-gray-400"
              >
                {isGenerating ? 'Generating...' : 'Generate Content'}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ModuleQuestionnaireModal;
