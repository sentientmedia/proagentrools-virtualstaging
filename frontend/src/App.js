import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Header Component
const Header = () => (
  <header className="bg-gradient-to-r from-blue-900 to-blue-700 text-white shadow-lg">
    <div className="container mx-auto px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-blue-400 rounded-lg flex items-center justify-center">
            <span className="text-xl font-bold">PA</span>
          </div>
          <h1 className="text-2xl font-bold">ProAgentTools</h1>
        </div>
        <nav className="hidden md:flex space-x-6">
          <a href="#interior-design" className="hover:text-blue-300 transition-colors">Interior Design</a>
          <a href="#ai-tools" className="hover:text-blue-300 transition-colors">AI Tools</a>
          <a href="#pricing" className="hover:text-blue-300 transition-colors">Pricing</a>
        </nav>
      </div>
    </div>
  </header>
);

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

  const pollStatus = async (jobId) => {
    try {
      const response = await axios.get(`${API}/interior-design/status/${jobId}`);
      const status = response.data;
      
      if (status.status === 'completed') {
        setProcessedImage(status);
        setProcessing(false);
        setStatusMessage('');
        setCurrentJobId(null);
      } else if (status.status === 'failed') {
        setError(status.error_message || 'Processing failed');
        setProcessing(false);
        setStatusMessage('');
        setCurrentJobId(null);
      } else if (status.status === 'processing' || status.status === 'submitted') {
        setStatusMessage(status.message || 'Processing in progress...');
        // Poll again in 10 seconds
        setTimeout(() => pollStatus(jobId), 10000);
      }
    } catch (err) {
      console.error('Status check failed:', err);
      // Try again in 15 seconds if status check fails
      setTimeout(() => pollStatus(jobId), 15000);
    }
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setProcessing(true);
    setError(null);
    setProcessedImage(null);
    setStatusMessage('Uploading and starting AI processing...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API}/interior-design/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Start polling for status
      const jobId = response.data.id;
      setCurrentJobId(jobId);
      setStatusMessage('Image submitted! Processing started - this may take 2-3 minutes due to AI model startup...');
      
      // Start polling status
      setTimeout(() => pollStatus(jobId), 5000); // Start checking after 5 seconds

    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process image');
      setProcessing(false);
      setStatusMessage('');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    maxFiles: 1,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  return (
    <section id="interior-design" className="py-16 bg-white">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">AI Interior Design Enhancement</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload any interior photo and watch our AI transform it into a stunning, professionally designed space.
            Perfect for staging properties and showing potential to clients.
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-8">
            {/* Upload Area */}
            <div className="space-y-6">
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer
                  ${dragActive || isDragActive 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                  }`}
              >
                <input {...getInputProps()} />
                <div className="space-y-4">
                  <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
                    <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                    </svg>
                  </div>
                  <div>
                    <p className="text-lg font-medium text-gray-900">
                      {processing ? 'Processing...' : 'Drop your interior photo here'}
                    </p>
                    <p className="text-gray-500">or click to browse</p>
                  </div>
                </div>
              </div>

              {processing && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <div className="animate-spin w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full mt-0.5"></div>
                    <div className="flex-1">
                      <span className="text-blue-800 font-medium">AI Processing in Progress</span>
                      <p className="text-blue-600 text-sm mt-1">
                        {statusMessage || 'Your custom interior design AI is starting up and processing your image...'}
                      </p>
                      <p className="text-blue-500 text-xs mt-2">
                        ⏱️ Estimated time: 2-3 minutes (due to AI model cold boot)
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <p className="text-red-800">{error}</p>
                </div>
              )}
            </div>

            {/* Results Area */}
            <div className="space-y-6">
              {processedImage ? (
                <div className="bg-gray-50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Enhanced Interior Design</h3>
                  <div className="space-y-4">
                    <img
                      src={processedImage.processed_image_url}
                      alt="AI Enhanced Interior"
                      className="w-full rounded-lg shadow-lg"
                      onLoad={() => console.log('Image loaded successfully')}
                      onError={() => console.error('Failed to load processed image')}
                    />
                    <div className="flex justify-between items-center text-sm text-gray-600">
                      <span>Original: {processedImage.original_filename}</span>
                      <span className="bg-green-100 text-green-800 px-2 py-1 rounded">✅ Completed</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-gray-100 rounded-xl p-6 h-64 flex items-center justify-center">
                  <div className="text-center">
                    <p className="text-gray-500 mb-2">Upload an image to see the AI-enhanced result</p>
                    {processing && (
                      <p className="text-blue-600 text-sm">Your image will appear here when processing completes</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
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