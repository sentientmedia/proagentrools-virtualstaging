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

  const modules = [
    // Foundation modules (auto-generated, always unlocked)
    { id: 'neighborhood_research', name: 'Neighborhood Research', icon: '🏘️', ai: true, credits: 0, description: 'Detailed neighborhood analysis', isFoundation: true, category: 'Foundation' },
    { id: 'listing_copy', name: 'Property Description', icon: '✍️', ai: true, credits: 0, description: 'Professional listing copy', isFoundation: true, category: 'Foundation' },
    { id: 'market_intel', name: 'Market Intelligence', icon: '📊', ai: true, credits: 0, description: 'Strategic market positioning', isFoundation: true, category: 'Foundation' },
    
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
    
    // Media
    { id: 'images', name: 'Photos & Interior Design', icon: '📸', description: 'Upload and enhance images', category: 'Media' },
  ];

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
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="max-w-none whitespace-pre-wrap text-gray-900 leading-relaxed">
                {content.content.replace(/\*\*/g, '')}
              </div>
            </div>
            <div className="flex items-center justify-between mt-2">
              <div className="text-xs text-gray-500">
                {content.is_ai_generated ? '🤖 AI Generated' : '✏️ Manually Edited'} • 
                Last updated: {new Date(content.last_edited || content.generated_at).toLocaleString()}
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
                  onClick={() => handleGenerateContent(module.id)}
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

        {/* Designer Gallery */}
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Choose Your Designer</h2>
          <p className="text-gray-600 mb-6">Select from 12 award-winning interior designers • Double-click to read full bio</p>
          <div className="grid md:grid-cols-3 lg:grid-cols-4 gap-4">
            {designers.map(designer => (
              <div 
                key={designer.id} 
                className="bg-white border-2 border-gray-200 rounded-lg overflow-hidden hover:border-blue-400 hover:shadow-lg transition-all cursor-pointer"
                onDoubleClick={() => {
                  setSelectedDesigner(designer);
                  setShowDesignerModal(true);
                }}
              >
                <div className="h-48 overflow-hidden">
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
              </div>
            ))}
          </div>
        </div>

        {/* Color Schemes */}
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Color Schemes</h2>
          <p className="text-gray-600 mb-6">20 carefully curated palettes to match any style and mood</p>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {colorSchemes.map(scheme => (
              <div key={scheme.id} className="bg-white border-2 border-gray-200 rounded-lg p-3 hover:border-blue-400 hover:shadow-lg transition-all cursor-pointer">
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
              </div>
            ))}
          </div>
        </div>

        {/* Upload Section */}
        <div className="border-t border-gray-200 pt-8">
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
              <span className="text-sm text-gray-600">Select multiple images to upload at once</span>
            </div>
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

        {/* AI Interior Design Section */}
        {images.length > 0 && (
          <div className="border-t border-gray-200 pt-6">
            {/* Info Banner */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <p className="text-sm text-blue-800">
                🎨 <strong>AI Interior Design:</strong> Enter your design preferences below, then select images and set room types. Process with AI (5 credits per image)
              </p>
            </div>

            {/* Design Preferences */}
            <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Design Preferences</h3>
              <p className="text-sm text-gray-600 mb-4">Describe the interior design style and colors you want for all selected images</p>
              
              <div className="space-y-4">
                {/* Design Description */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Design Style Description
                  </label>
                  <textarea
                    value={customDescription}
                    onChange={(e) => setCustomDescription(e.target.value)}
                    placeholder="e.g., Modern coastal vibes with natural textures and airy atmosphere, Scandinavian minimalism with warm woods..."
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg text-sm resize-none"
                    rows="3"
                  />
                </div>

                {/* Color Preferences */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Color Palette
                  </label>
                  <input
                    type="text"
                    value={customColors}
                    onChange={(e) => setCustomColors(e.target.value)}
                    placeholder="e.g., Soft sage green, warm beige, ivory white, natural wood tones..."
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg text-sm"
                  />
                </div>
              </div>
            </div>

            {/* Action Header */}
            <div className="flex items-center justify-between mb-6">
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
                      <div className="p-3 bg-white">
                        <div className="text-xs text-gray-600">
                          {variant.room_type} • {variant.designer}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          {variant.status === 'completed' ? '✓ Completed' : '⏳ Processing...'}
                        </div>
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
        )}
      </div>
    );
  };

  // renderInteriorDesign function removed - functionality integrated into renderImages

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
    
    const foundationComplete = listing.foundation_status === 'completed';
    const foundationProcessing = listing.foundation_status === 'processing';
    const foundationModules = modules.filter(m => m.isFoundation);
    const dependentModules = modules.filter(m => m.requiresFoundation);
    const otherModules = modules.filter(m => !m.isFoundation && !m.requiresFoundation && !m.ai);

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
                  ✅ {Object.keys(moduleContent).length} content pieces
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
                    if (!isLocked) {
                      setActiveModule(module.id);
                      setActiveView('module');
                    }
                  }}
                  disabled={isLocked}
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
              <div className="text-center py-20">
                <div className="w-32 h-32 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <span className="text-6xl">🏠</span>
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  {listing.property_details.address}
                </h2>
                <p className="text-lg text-gray-600 mb-2">
                  {listing.property_details.beds} bed • {listing.property_details.baths} bath • 
                  {listing.property_details.sqft && ` ${listing.property_details.sqft.toLocaleString()} sq ft`}
                </p>
                {listing.property_details.listing_price && (
                  <p className="text-3xl font-bold text-blue-600 mb-8">
                    ${listing.property_details.listing_price.toLocaleString()}
                  </p>
                )}
                <p className="text-gray-600 mb-4">
                  Select a tool from the left sidebar to get started
                </p>
                {foundationProcessing && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 inline-block">
                    <p className="text-sm text-blue-800">
                      Foundation content is being generated. This takes 1-2 minutes.
                    </p>
                  </div>
                )}
              </div>
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
    </div>
  );
};

export default IndividualListingPage;
