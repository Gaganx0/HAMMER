import os
import json
import requests
from time import sleep

API_KEY = 'YOUR_BING_API_KEY'
ENDPOINT = 'https://api.bing.microsoft.com/v7.0/images/visualsearch'
IMAGE_FOLDER = 'restored_images'  # Folder with your 250 images
OUTPUT_JSON = 'reverse_search_results.json'

HEADERS = {'Ocp-Apim-Subscription-Key': API_KEY}

results = {}

for image_file in os.listdir(IMAGE_FOLDER):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(IMAGE_FOLDER, image_file)
    with open(image_path, 'rb') as f:
        files = {'image': (image_file, f)}
        try:
            response = requests.post(ENDPOINT, headers=HEADERS, files=files)
            data = response.json()

            # Attempt to extract top visually similar image + source
            tags = data.get('tags', [])
            if tags:
                actions = tags[0].get('actions', [])
                for action in actions:
                    if action['actionType'] == 'VisualSearch':
                        first_match = action.get('data', {}).get('value', [])[0]
                        match_url = first_match.get('contentUrl')
                        source_page = first_match.get('hostPageUrl')

                        results[image_file] = {
                            'match_url': match_url,
                            'source_page': source_page,
                            'status': 'match_found'
                        }
                        break
                else:
                    results[image_file] = {'status': 'no_exact_match'}
            else:
                results[image_file] = {'status': 'no_tags_found'}

        except Exception as e:
            results[image_file] = {'status': 'error', 'error': str(e)}

    sleep(1)  # Respectful delay to avoid rate limiting

# Save results
with open(OUTPUT_JSON, 'w') as out_f:
    json.dump(results, out_f, indent=4)
