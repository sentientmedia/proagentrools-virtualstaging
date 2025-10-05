import React, { useState } from 'react';

const CreditConfirmationModal = ({ 
  isOpen, 
  onConfirm, 
  onCancel, 
  totalCredits, 
  selectedTools = [],
  actionType = "process", // "process" or "create"
  propertyAddress = ""
}) => {
  const [dontAskAgain, setDontAskAgain] = useState(false);

  if (!isOpen) return null;

  const handleConfirm = () => {
    // Save "don't ask again" preference to localStorage if checked
    if (dontAskAgain) {
      localStorage.setItem('creditConfirmationDisabled', 'true');
    }
    
    onConfirm();
  };

  const getActionText = () => {
    switch (actionType) {
      case "create":
        return "create this listing with selected AI tools";
      case "process":
        return "process the selected AI tools for this listing";
      default:
        return "proceed with this action";
    }
  };

  const getToolsByCategory = () => {
    const categories = {};
    selectedTools.forEach(tool => {
      const category = tool.category || 'Other';
      if (!categories[category]) {
        categories[category] = [];
      }
      categories[category].push(tool);
    });
    return categories;
  };

  const toolsByCategory = getToolsByCategory();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
        
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
              <span className="text-2xl">💳</span>
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Confirm Credit Usage</h2>
              <p className="text-gray-600">Review the cost before proceeding</p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          
          {/* Property Info */}
          {propertyAddress && (
            <div className="bg-blue-50 rounded-lg p-4 mb-6">
              <h3 className="font-medium text-blue-900 mb-1">Property</h3>
              <p className="text-blue-800">{propertyAddress}</p>
            </div>
          )}

          {/* Credit Cost Summary */}
          <div className="bg-gray-50 rounded-lg p-4 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Cost Summary</h3>
              <div className="text-right">
                <div className="text-2xl font-bold text-blue-600">{totalCredits}</div>
                <div className="text-sm text-gray-600">credits</div>
              </div>
            </div>
            
            <div className="text-sm text-gray-600 mb-4">
              This will {getActionText()} using {totalCredits} credits from your account.
            </div>

            {/* Selected Tools Breakdown */}
            {selectedTools.length > 0 && (
              <div>
                <h4 className="font-medium text-gray-900 mb-3">
                  Selected Tools ({selectedTools.length})
                </h4>
                
                <div className="space-y-4 max-h-48 overflow-y-auto">
                  {Object.entries(toolsByCategory).map(([category, tools]) => (
                    <div key={category}>
                      <h5 className="text-sm font-medium text-gray-700 mb-2 flex items-center space-x-2">
                        <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                        <span>{category}</span>
                      </h5>
                      <div className="ml-4 space-y-1">
                        {tools.map(tool => (
                          <div key={tool.tool_id || tool.id} className="flex items-center justify-between text-sm">
                            <span className="text-gray-700">{tool.tool_name || tool.name}</span>
                            <span className="text-gray-500 font-medium">
                              {tool.credits_cost || tool.cost} credits
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Credit Balance Warning */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <div className="flex items-start space-x-3">
              <span className="text-yellow-600 text-xl">⚠️</span>
              <div>
                <h4 className="font-medium text-yellow-800 mb-1">Credit Usage</h4>
                <p className="text-yellow-700 text-sm">
                  Make sure you have sufficient credits in your account. 
                  You can purchase more credits from your dashboard if needed.
                </p>
              </div>
            </div>
          </div>

          {/* Don't Ask Again Option */}
          <div className="border border-gray-200 rounded-lg p-4">
            <label className="flex items-start space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={dontAskAgain}
                onChange={(e) => setDontAskAgain(e.target.checked)}
                className="mt-0.5 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <div>
                <span className="text-sm font-medium text-gray-900">
                  Don't ask me again
                </span>
                <p className="text-xs text-gray-500 mt-1">
                  Future AI tool processing will proceed without this confirmation dialog. 
                  You can re-enable confirmations in your account settings.
                </p>
              </div>
            </label>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t flex items-center justify-between">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors font-medium"
          >
            Cancel
          </button>
          
          <div className="flex items-center space-x-3">
            <div className="text-right text-sm text-gray-600">
              <div>Total Cost: <span className="font-semibold text-gray-900">{totalCredits} credits</span></div>
            </div>
            <button
              onClick={handleConfirm}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold"
            >
              Confirm & Continue
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};

export default CreditConfirmationModal;