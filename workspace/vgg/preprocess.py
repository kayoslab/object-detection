# Configuration file for the wound-test project
from configuration import config

from imutils import paths

import os
import shutil
import xmltodict


# Initialises a flat list of all labels as a helper set to speed up the process
labels = set()
# Initialises a list of dictionaries to store the classified data
data = []

# loop over all CSV files in the annotations directory
for xmlPath in paths.list_files(config.LABELLED_PATH, validExts=(".xml")):
    # Open the file and read the contents
    with open(xmlPath, 'r', encoding='utf-8') as file:
        xml_file = file.read()
        # Parse the XML file into a dictionary
        xml_dict = xmltodict.parse(xml_file)
        # Extract the filename, bounding box coordinates, and class label
        xml_filename = xml_dict['annotation']['filename']
        for element in xml_dict['annotation']['object']:
            if element['name'] not in labels:
                labels.add(element['name'])
                data.append(
                    {
                        'label': element['name'], 
                        'data': [
                            {
                                'file': xml_filename,
                                'label': element['name'],
                                'bndbox': element['bndbox']
                            }
                        ]
                    }
                )
            else:
                for entry in data:
                    if entry['label'] == element['name']:
                        entry['data'].append(
                            {
                                'file': xml_filename,
                                'label': element['name'],
                                'bndbox': element['bndbox']
                            }
                        )
                        break

for label in data:
    # Check if folder exists
    if not os.path.exists(config.PROCESSED_ANNOTS_PATH):
        os.makedirs(config.PROCESSED_ANNOTS_PATH)
    
    if not os.path.exists(os.path.join(config.PROCESSED_IMAGES_PATH, label['label'])):
        os.makedirs(os.path.join(config.PROCESSED_IMAGES_PATH, label['label']))

    # Write to file
    with open(os.path.join(config.PROCESSED_ANNOTS_PATH, label['label'] + '.csv'), 'w') as file:
        for entry in label['data']:
            file.write(f"{entry['file']},{entry['bndbox']['xmin']},{entry['bndbox']['ymin']},{entry['bndbox']['xmax']},{entry['bndbox']['ymax']},{entry['label']}\n")            
            shutil.copy2(os.path.join(config.LABELLED_PATH, entry['file']), os.path.join(config.PROCESSED_IMAGES_PATH, entry['label'], entry['file']))
