import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import axios from 'axios';

const AIResultsModal = ({ listingId, onClose }) => {
  const { token } = useAuth();
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');

  useEffect(() => {
    if (listingId) {
      loadAIResults();
    }
  }, [listingId]);

  const loadAIResults = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${BACKEND_URL}/api/listings/${listingId}/ai-results`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      const aiResults = response.data.ai_results;
      setResults(aiResults);
      
      // Set first category as selected by default
      if (aiResults?.outputs) {
        const categories = Object.keys(aiResults.outputs).filter(key => key !== 'unified_summary');
        if (categories.length > 0) {
          setSelectedCategory(categories[0]);
        }
      }
      
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load AI results');
    } finally {
      setLoading(false);
    }
  };

  const formatToolOutput = (toolData) => {
    if (!toolData || !toolData.output) return 'No output available';

    const output = toolData.output;
    
    // If output has structured data, format it nicely
    if (typeof output === 'object' && output.content) {
      // Try to parse content if it's JSON string
      if (typeof output.content === 'string') {
        try {
          const parsed = JSON.parse(output.content);
          return formatParsedOutput(parsed);
        } catch {
          return output.content;
        }
      }
      return output.content;
    }
    
    // If output is JSON string, try to parse and format
    if (typeof output === 'string') {
      try {
        const parsed = JSON.parse(output);
        return formatParsedOutput(parsed);
      } catch {
        return output;
      }
    }
    
    // If output is already an object, format it
    if (typeof output === 'object') {
      return formatParsedOutput(output);
    }
    
    return String(output);
  };

  const formatParsedOutput = (data) => {
    if (!data || typeof data !== 'object') return String(data);

    let formatted = '';

    // Handle different output structures
    Object.entries(data).forEach(([key, value]) => {
      if (key === 'TOOL 1' || key.includes('TOOL')) {
        // Skip tool wrapper keys
        if (typeof value === 'object') {
          formatted += formatParsedOutput(value);
        }
        return;
      }

      // Format the key nicely
      const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      
      if (Array.isArray(value)) {
        formatted += `**${formattedKey}:**\n`;
        value.forEach((item, index) => {
          formatted += `${index + 1}. ${item}\n`;
        });
        formatted += '\n';
      } else if (typeof value === 'object') {
        formatted += `**${formattedKey}:**\n`;
        formatted += formatParsedOutput(value);
        formatted += '\n';
      } else {
        formatted += `**${formattedKey}:**\n${value}\n\n`;
      }
    });

    return formatted;
  };

  const renderFormattedOutput = (text) => {
    // Convert markdown-style formatting to JSX
    const lines = text.split('\n');
    const elements = [];
    
    lines.forEach((line, index) => {
      if (line.startsWith('**') && line.endsWith(':**')) {
        // Header
        const headerText = line.slice(2, -3);
        elements.push(
          <h4 key={index} className="font-semibold text-gray-900 mt-4 mb-2">
            {headerText}
          </h4>
        );
      } else if (line.match(/^\d+\./)) {
        // List item
        elements.push(
          <p key={index} className="ml-4 mb-1 text-gray-700">
            {line}
          </p>
        );
      } else if (line.trim()) {
        // Regular paragraph
        elements.push(
          <p key={index} className="mb-2 text-gray-700">
            {line}
          </p>
        );
      }
    });
    
    return elements.length > 0 ? elements : (
      <pre className="whitespace-pre-wrap text-sm text-gray-700 font-sans">
        {text}
      </pre>
    );
  };

  const getCategoryIcon = (category) => {
    const icons = {
      'Marketing & Creative': '🎨',
      'Staging & Design': '🏠',
      'Due-Diligence & Compliance': '📋',
      'Market Intel & Strategy': '📊',
      'Process & Productivity': '⚡'
    };
    return icons[category] || '🔧';
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span>Loading AI results...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-lg p-6 max-w-md w-full">
          <h3 className="text-lg font-semibold text-red-900 mb-2">Error Loading Results</h3>
          <p className="text-red-700 mb-4">{error}</p>
          <div className="flex justify-end space-x-3">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
            >
              Close
            </button>
            <button
              onClick={loadAIResults}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return null;
  }

  const categories = Object.keys(results.outputs || {}).filter(key => key !== 'unified_summary');
  const unifiedSummary = results.outputs?.unified_summary;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-6xl w-full h-[80vh] flex flex-col">
        
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">AI Processing Results</h2>
            <p className="text-gray-600">
              {results.tools_processed} tools processed • {new Date(results.processed_at).toLocaleString()}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">
          
          {/* Sidebar - Categories */}
          <div className="w-64 bg-gray-50 border-r overflow-y-auto">
            <div className="p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Tool Categories</h3>
              
              {/* Unified Summary */}
              {unifiedSummary && (
                <button
                  onClick={() => setSelectedCategory('unified_summary')}
                  className={`w-full text-left p-3 rounded-lg mb-2 transition-colors ${
                    selectedCategory === 'unified_summary'
                      ? 'bg-blue-100 text-blue-900 border-blue-200'
                      : 'bg-white hover:bg-gray-100 text-gray-700'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span>📋</span>
                    <span className="font-medium">Executive Summary</span>
                  </div>
                </button>
              )}
              
              {/* Tool Categories */}
              {categories.map(category => {
                const categoryData = results.outputs[category];
                const toolCount = Object.keys(categoryData || {}).length;
                
                return (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category)}
                    className={`w-full text-left p-3 rounded-lg mb-2 transition-colors ${
                      selectedCategory === category
                        ? 'bg-blue-100 text-blue-900 border-blue-200'
                        : 'bg-white hover:bg-gray-100 text-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span>{getCategoryIcon(category)}</span>
                        <span className="font-medium text-sm">{category}</span>
                      </div>
                      <span className="text-xs bg-gray-200 px-2 py-1 rounded">
                        {toolCount}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-6">
              
              {selectedCategory === 'unified_summary' && unifiedSummary ? (
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">
                    📋 Executive Summary
                  </h3>
                  <div className="bg-blue-50 rounded-lg p-4 mb-6">
                    <pre className="whitespace-pre-wrap text-gray-800 font-sans">
                      {unifiedSummary.summary}
                    </pre>
                  </div>
                </div>
              ) : selectedCategory && results.outputs[selectedCategory] ? (
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                    <span>{getCategoryIcon(selectedCategory)}</span>
                    <span>{selectedCategory}</span>
                  </h3>
                  
                  <div className="space-y-6">
                    {Object.entries(results.outputs[selectedCategory]).map(([toolId, toolData]) => (
                      <div key={toolId} className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-3">
                          <h4 className="font-medium text-gray-900">
                            {toolData.tool_name || toolId}
                          </h4>
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            toolData.status === 'completed' 
                              ? 'bg-green-100 text-green-800'
                              : toolData.status === 'error'
                              ? 'bg-red-100 text-red-800'
                              : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {toolData.status}
                          </span>
                        </div>
                        
                        {toolData.status === 'completed' ? (
                          <div className="bg-white rounded border p-4">
                            <div className="prose prose-sm max-w-none">
                              {renderFormattedOutput(formatToolOutput(toolData))}
                            </div>
                          </div>
                        ) : toolData.status === 'error' ? (
                          <div className="bg-red-50 border border-red-200 rounded p-3">
                            <p className="text-red-800 text-sm">
                              Error: {toolData.error || 'Processing failed'}
                            </p>
                          </div>
                        ) : (
                          <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
                            <p className="text-yellow-800 text-sm">
                              Status: {toolData.status}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-gray-400 text-6xl mb-4">🤖</div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Select a category to view results
                  </h3>
                  <p className="text-gray-600">
                    Choose a tool category from the sidebar to see detailed AI-generated content.
                  </p>
                </div>
              )}
              
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t px-6 py-4 bg-gray-50">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              Processing ID: {results.processing_id}
            </div>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Close
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};

export default AIResultsModal;