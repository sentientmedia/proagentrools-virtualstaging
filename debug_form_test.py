#!/usr/bin/env python3

import requests
from PIL import Image
import io

def test_form_parameters():
    """Test if form parameters are being sent correctly"""
    
    # Create test image
    img = Image.new('RGB', (100, 100), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)

    # Test with custom parameters
    files = {'file': ('test_interior.jpg', img_buffer, 'image/jpeg')}
    data = {
        'room_type': 'bedroom',
        'designer': 'adrian_mercer',
        'color_scheme': 'nomad_prism'
    }

    url = 'https://7f9f2de6-2fb0-4a74-af1b-8d15dbd9c892.preview.emergentagent.com/api/interior-design/process'
    
    print("Sending request with:")
    print(f"  room_type: {data['room_type']}")
    print(f"  designer: {data['designer']}")
    print(f"  color_scheme: {data['color_scheme']}")
    
    response = requests.post(url, data=data, files=files)
    
    print(f"\nResponse status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response values:")
        print(f"  room_type: {result.get('room_type')}")
        print(f"  designer: {result.get('designer')}")
        print(f"  color_scheme: {result.get('color_scheme')}")
        
        # Check if values match
        if (result.get('room_type') == data['room_type'] and 
            result.get('designer') == data['designer'] and 
            result.get('color_scheme') == data['color_scheme']):
            print("\n✅ Custom parameters working correctly!")
            return True
        else:
            print("\n❌ Custom parameters not working - using defaults instead")
            return False
    else:
        print(f"Error: {response.text}")
        return False

if __name__ == "__main__":
    test_form_parameters()