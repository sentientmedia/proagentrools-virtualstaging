#!/usr/bin/env python3
"""
ProAgentTools MCP Mega-Agent Server
Unified AI agent that processes all 30 real estate tools for listings
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
from emergentintegrations import get_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProAgentToolsMegaAgent:
    """
    Unified AI agent that processes all real estate tools using emergent integrations
    """
    
    def __init__(self):
        self.client = get_client()
        self.session_id = str(uuid.uuid4())
        
    async def process_listing_tools(self, listing_data: Dict[str, Any], selected_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process all selected AI tools for a listing in a single cohesive operation
        
        Args:
            listing_data: Complete listing information including property details
            selected_tools: List of AI tools selected for processing
            
        Returns:
            Dict containing all processed tool outputs
        """
        try:
            logger.info(f"Starting mega-agent processing for {len(selected_tools)} tools")
            
            # Create unified context from listing data
            context = self._build_listing_context(listing_data)
            
            # Group tools by category for efficient processing
            tools_by_category = self._group_tools_by_category(selected_tools)
            
            # Process each category with specialized prompts
            all_outputs = {}
            
            for category, tools in tools_by_category.items():
                logger.info(f"Processing {category} tools: {[t['tool_name'] for t in tools]}")
                category_output = await self._process_category_tools(context, category, tools)
                all_outputs[category] = category_output
            
            # Generate unified summary and recommendations
            unified_summary = await self._generate_unified_summary(context, all_outputs)
            all_outputs['unified_summary'] = unified_summary
            
            return {
                'success': True,
                'processing_id': self.session_id,
                'processed_at': datetime.utcnow().isoformat(),
                'tools_processed': len(selected_tools),
                'outputs': all_outputs
            }
            
        except Exception as e:
            logger.error(f"Mega-agent processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_id': self.session_id
            }
    
    def _build_listing_context(self, listing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive context from listing data"""
        property_details = listing_data.get('property_details', {})
        
        context = {
            'property_address': f"{property_details.get('address')}, {property_details.get('city')}, {property_details.get('state')} {property_details.get('zip_code')}",
            'property_type': property_details.get('property_type'),
            'beds': property_details.get('beds'),
            'baths': property_details.get('baths'),
            'sqft': property_details.get('sqft'),
            'year_built': property_details.get('year_built'),
            'listing_price': property_details.get('listing_price'),
            'mls_number': property_details.get('mls_number'),
            'description': listing_data.get('description', ''),
            'agent_notes': listing_data.get('agent_notes', ''),
            'listing_id': listing_data.get('id'),
            'processing_timestamp': datetime.utcnow().isoformat()
        }
        
        return context
    
    def _group_tools_by_category(self, selected_tools: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group tools by category for efficient batch processing"""
        categories = {}
        for tool in selected_tools:
            category = tool.get('category', 'Unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(tool)
        return categories
    
    async def _process_category_tools(self, context: Dict[str, Any], category: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process all tools in a specific category"""
        
        # Create category-specific system prompt
        system_prompt = self._get_category_system_prompt(category, context)
        
        # Create tool-specific prompts
        tool_prompts = []
        for tool in tools:
            tool_prompt = self._get_tool_specific_prompt(tool, context)
            tool_prompts.append({
                'tool_id': tool['tool_id'],
                'tool_name': tool['tool_name'],
                'prompt': tool_prompt
            })
        
        # Process with unified context
        try:
            response = await self.client.achat.completions.create(
                model="openai/gpt-4o",  # Using latest GPT-4 for best results
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self._format_batch_request(tool_prompts)}
                ],
                max_tokens=4000,
                temperature=0.7
            )
            
            # Parse structured response
            content = response.choices[0].message.content
            parsed_outputs = self._parse_batch_response(content, tools)
            
            return parsed_outputs
            
        except Exception as e:
            logger.error(f"Category processing failed for {category}: {str(e)}")
            return {'error': str(e), 'tools': [t['tool_name'] for t in tools]}
    
    def _get_category_system_prompt(self, category: str, context: Dict[str, Any]) -> str:
        """Get specialized system prompt for each category"""
        
        base_context = f"""
Property: {context['property_address']}
Type: {context['property_type']} | {context['beds']} bed, {context['baths']} bath | {context['sqft']} sqft
Price: ${context['listing_price']:,} | MLS: {context['mls_number']}
Year Built: {context['year_built']}
"""
        
        category_prompts = {
            "Marketing & Creative": f"""
You are an expert real estate marketing specialist. You create compelling, SEO-optimized marketing content that converts leads.

{base_context}

Your task is to create professional marketing materials that highlight the property's best features, target the right audience, and drive engagement. Focus on:
- Compelling headlines and descriptions
- Social media content that gets shares
- SEO optimization
- Emotional connection with buyers
- Clear calls-to-action
""",
            
            "Staging & Design": f"""
You are a professional home staging consultant and interior designer specializing in real estate marketing.

{base_context}

Your task is to provide staging recommendations that maximize the property's appeal and perceived value. Focus on:
- Furniture placement and selection
- Color schemes and lighting
- Decluttering strategies  
- Budget-conscious improvements
- Target buyer lifestyle alignment
""",
            
            "Due-Diligence & Compliance": f"""
You are a real estate compliance and due diligence expert with deep knowledge of regulations, permits, and property law.

{base_context}

Your task is to identify potential issues, ensure compliance, and provide clear guidance on legal and regulatory matters. Focus on:
- Fair housing compliance
- Permit and zoning analysis
- Risk assessment
- Legal documentation review
- Regulatory compliance checks
""",
            
            "Market Intel & Strategy": f"""
You are a real estate market analyst and strategic advisor with expertise in property valuation and market trends.

{base_context}

Your task is to provide data-driven insights and strategic recommendations for pricing, positioning, and market approach. Focus on:
- Comparative market analysis
- Pricing strategy
- Market positioning
- Investment analysis
- Demographic insights
""",
            
            "Process & Productivity": f"""
You are a real estate operations expert focused on streamlining processes and improving agent productivity.

{base_context}

Your task is to create systematic approaches, templates, and workflows that save time and improve client experience. Focus on:
- Process automation
- Client communication
- Event planning
- Task management
- Quality assurance checklists
"""
        }
        
        return category_prompts.get(category, f"You are a real estate professional working on: {category}")
    
    def _get_tool_specific_prompt(self, tool: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate specific prompt for each tool"""
        
        tool_id = tool.get('tool_id', '')
        tool_name = tool.get('tool_name', '')
        description = tool.get('description', '')
        
        # Base property info
        property_info = f"""
Property Details:
- Address: {context['property_address']}
- Type: {context['property_type']}
- Bedrooms: {context['beds']} | Bathrooms: {context['baths']} 
- Square Feet: {context['sqft']} | Year Built: {context['year_built']}
- Listing Price: ${context['listing_price']:,}
- MLS Number: {context['mls_number']}
- Current Description: {context['description']}
- Agent Notes: {context['agent_notes']}
"""
        
        # Tool-specific prompts based on the catalog
        tool_prompts = {
            "listing_luxe_gpt": f"""
{property_info}

Create a luxurious, MLS-ready property description (300 words max) that includes:
1. An attention-grabbing 60-character headline
2. SEO-optimized main description highlighting unique features
3. Three social media post snippets for Instagram/Facebook
4. Emotional language that creates desire
5. Call-to-action phrases

Format as JSON with keys: headline, description, social_snippets[]
""",
            
            "social_snippets_studio": f"""
{property_info}

Create a week's worth of social media content including:
1. 5 Instagram captions with hooks and CTAs
2. 3 Instagram Reels hook scripts (15-second openers)  
3. 10 hashtag clusters for different platforms
4. Facebook post variations
5. LinkedIn professional post

Target audience: Potential homebuyers and real estate network
Format as JSON with keys: ig_captions[], reels_hooks[], hashtag_clusters[], facebook_posts[], linkedin_post
""",
            
            "photofix_wizard": f"""
{property_info}

Analyze typical listing photo issues and provide:
1. Common photography problems to check for
2. Retouching recommendations by room type
3. Stable Diffusion prompts for photo enhancement
4. Before/after improvement suggestions
5. Professional photography tips

Focus on maximizing visual appeal and buyer engagement.
Format as JSON with keys: photo_issues[], retouching_notes[], sd_prompts[], improvement_tips[]
""",
            
            "comp_cruncher_cma": f"""
{property_info}

Provide a comprehensive market analysis including:
1. Pricing strategy based on property type and location
2. Market positioning recommendations
3. Days on market expectations
4. Competitive advantages to highlight
5. Pricing flexibility recommendations

Create talking points for seller consultation.
Format as JSON with keys: price_range, market_position, dom_estimate, advantages[], talking_points[]
""",
            
            "open_house_orchestrator": f"""
{property_info}

Plan a comprehensive open house event including:
1. Invitation copy for different channels
2. SMS reminder templates  
3. QR code sign-in form setup
4. Follow-up email sequence (3 emails)
5. Event timeline and checklist

Target the right buyer demographic for this property type.
Format as JSON with keys: invitation_copy, sms_templates[], signin_qr_setup, followup_emails[], event_checklist[]
"""
        }
        
        # Return specific prompt or generic one
        return tool_prompts.get(tool_id, f"""
{property_info}

Tool: {tool_name}
Description: {description}

Please provide comprehensive output for this tool that helps real estate agents with this specific functionality.
Format as structured JSON with relevant keys for the tool's purpose.
""")
    
    def _format_batch_request(self, tool_prompts: List[Dict[str, Any]]) -> str:
        """Format multiple tool requests into a single batch prompt"""
        
        request_text = "Please process the following real estate tools for this property listing:\n\n"
        
        for i, tool_prompt in enumerate(tool_prompts, 1):
            request_text += f"""
=== TOOL {i}: {tool_prompt['tool_name']} ===
{tool_prompt['prompt']}

"""
        
        request_text += """
IMPORTANT: 
- Provide comprehensive, professional output for each tool
- Use JSON format where specified  
- Ensure all content is ready-to-use by real estate agents
- Maintain consistency across all outputs for this property
- Each tool output should be clearly labeled and separated

Please process all tools and provide structured outputs."""
        
        return request_text
    
    def _parse_batch_response(self, content: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse the AI response into structured tool outputs"""
        
        parsed_outputs = {}
        
        try:
            # Try to extract tool-specific sections
            for tool in tools:
                tool_name = tool['tool_name']
                tool_id = tool['tool_id']
                
                # Look for tool-specific markers in content
                if tool_name.upper() in content.upper():
                    # Extract content for this tool
                    tool_output = self._extract_tool_content(content, tool_name)
                    parsed_outputs[tool_id] = {
                        'tool_name': tool_name,
                        'status': 'completed',
                        'output': tool_output,
                        'processed_at': datetime.utcnow().isoformat()
                    }
                else:
                    # Fallback: create basic output
                    parsed_outputs[tool_id] = {
                        'tool_name': tool_name,
                        'status': 'partial',
                        'output': {'content': content[:500] + '...', 'note': 'Extracted from general response'},
                        'processed_at': datetime.utcnow().isoformat()
                    }
            
            return parsed_outputs
            
        except Exception as e:
            logger.error(f"Failed to parse batch response: {str(e)}")
            # Return error status for all tools
            return {
                tool['tool_id']: {
                    'tool_name': tool['tool_name'],
                    'status': 'error',
                    'error': str(e),
                    'processed_at': datetime.utcnow().isoformat()
                } for tool in tools
            }
    
    def _extract_tool_content(self, content: str, tool_name: str) -> Dict[str, Any]:
        """Extract specific content for a tool from the full response"""
        
        # Try to find JSON blocks first
        import re
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        if json_matches:
            try:
                return json.loads(json_matches[0])
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract text sections
        lines = content.split('\n')
        tool_content = []
        capture = False
        
        for line in lines:
            if tool_name.upper() in line.upper() or '===' in line:
                capture = True
            elif capture and ('===' in line or 'TOOL' in line.upper()):
                break
            elif capture:
                tool_content.append(line)
        
        return {
            'content': '\n'.join(tool_content).strip(),
            'extracted_from': 'text_parsing'
        }
    
    async def _generate_unified_summary(self, context: Dict[str, Any], all_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a unified summary and action plan from all tool outputs"""
        
        try:
            summary_prompt = f"""
Based on all the AI tool outputs for this property listing, create a unified executive summary and action plan.

Property: {context['property_address']}
Tools Processed: {list(all_outputs.keys())}

Please provide:
1. Executive Summary (key insights across all categories)
2. Priority Action Items (top 5 recommendations)
3. Marketing Strategy Overview
4. Timeline and Next Steps
5. Budget Considerations

Focus on actionable insights that help the agent succeed with this listing.
"""
            
            response = await self.client.achat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior real estate strategist creating executive summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return {
                'summary': response.choices[0].message.content,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate unified summary: {str(e)}")
            return {
                'summary': 'Summary generation failed',
                'error': str(e)
            }

# MCP Server Integration
async def run_mega_agent_processing(listing_data: Dict[str, Any], selected_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main entry point for MCP mega-agent processing
    """
    agent = ProAgentToolsMegaAgent()
    result = await agent.process_listing_tools(listing_data, selected_tools)
    return result

if __name__ == "__main__":
    # Test the mega-agent
    test_listing = {
        'id': 'test-123',
        'property_details': {
            'address': '123 Test Street',
            'city': 'San Francisco',
            'state': 'CA',
            'zip_code': '94102',
            'beds': 3,
            'baths': 2.5,
            'sqft': 2000,
            'property_type': 'Single Family',
            'listing_price': 1200000,
            'mls_number': 'ML12345'
        },
        'description': 'Beautiful Victorian home',
        'agent_notes': 'Motivated seller'
    }
    
    test_tools = [
        {'tool_id': 'listing_luxe_gpt', 'tool_name': 'Listing-Luxe GPT', 'category': 'Marketing & Creative'},
        {'tool_id': 'social_snippets_studio', 'tool_name': 'Social-Snippets Studio', 'category': 'Marketing & Creative'}
    ]
    
    async def test():
        result = await run_mega_agent_processing(test_listing, test_tools)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())