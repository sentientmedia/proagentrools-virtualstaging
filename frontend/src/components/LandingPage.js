import React, { useState } from 'react';
import AuthModal from './auth/AuthModal';
import { useAuth } from '../contexts/AuthContext';

// Hero Section Component
const HeroSection = ({ onGetStarted }) => (
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
        <button 
          onClick={onGetStarted}
          className="bg-blue-600 text-white px-8 py-4 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
        >
          Get Started Free
        </button>
        <a href="#features" className="bg-white text-blue-600 border-2 border-blue-600 px-8 py-4 rounded-lg font-semibold hover:bg-blue-50 transition-colors">
          Explore AI Tools
        </a>
      </div>
    </div>
  </section>
);

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
    <section id="features" className="py-16 bg-white">
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
const PricingSection = ({ onGetStarted }) => (
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
            <button 
              onClick={onGetStarted}
              className={`w-full py-3 px-6 rounded-lg font-semibold transition-colors ${
                plan.popular 
                  ? 'bg-blue-600 text-white hover:bg-blue-700' 
                  : 'bg-gray-200 text-gray-900 hover:bg-gray-300'
              }`}
            >
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

// Main Landing Page Component
const LandingPage = () => {
  const [showAuthModal, setShowAuthModal] = useState(false);
  const { isAuthenticated } = useAuth();

  const handleGetStarted = () => {
    if (isAuthenticated) {
      // Redirect to dashboard or tools
      window.location.href = '/dashboard';
    } else {
      setShowAuthModal(true);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <HeroSection onGetStarted={handleGetStarted} />
      <FeaturesSection />
      <PricingSection onGetStarted={handleGetStarted} />
      <Footer />
      
      {showAuthModal && (
        <AuthModal 
          isOpen={showAuthModal} 
          onClose={() => setShowAuthModal(false)}
          initialMode="register"
        />
      )}
    </div>
  );
};

export default LandingPage;