## Basic pipeline that processes image to extract using Optical character Recognition (OCR)
# 1. Downloading and processing images.
# 2. Using OCR (Tesseract) to extract text.
# 3. Extracting numerical values and units based on pre-defined patterns.
# 4. Formatting the output to match the required format.

import pandas as pd 
import requests
import pytesseract
# Specify the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
from io import BytesIO
import re 




# Load the dataset
test_df = pd.read_csv("D:\\python projects\\files\\Amazon Ml\\sample_test.csv")

# Function to download and open an image from url.
def download_image(image_url):
    try:
        response = requests.get(image_url)
        img= Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f'Error downloading image: {e}')
        return None
    
# Function to extract text from an image using OCR
def extract_text_from_image(image):
    if image is not None:
        return pytesseract.image_to_string(image)
    return ""
    
# Function to extract values and units from text
def extract_value_and_unit(text, entity_name):
    # Define regex patterns fpr different entities 
    entity_unit_map={
        "width": r'(\d+\.?\d*)\s*(centimetre|foot|millimetre|metre|inch|yard)',
        "depth": r'(\d+\.?\d*)\s*(centimetre|foot|millimetre|metre|inch|yard)',
        "height": r'(\d+\.?\d*)\s*(centimetre|foot|millimetre|metre|inch|yard)',
        "item_weight": r'(\d+\.?\d*)\s*(milligram|kilogram|microgram|gram|ounce|ton|pound)',
        "maximum_weight_recommendation": r'(\d+\.?\d*)\s*(milligram|kilogram|microgram|gram|ounce|ton|pound)',
        "voltage": r'(\d+\.?\d*)\s*(millivolt|kilovolt|volt)',
        "wattage": r'(\d+\.?\d*)\s*(kilowatt|watt)',
        "item_volume": r'(\d+\.?\d*)\s*(cubic foot|microlitre|cup|fluid ounce|centilitre|imperial gallon|pint|decilitre|litre|millilitre|quart|cubic inch|gallon)'
    }
    pattern = entity_unit_map.get(entity_name, None)
    if pattern :
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value , unit = match.groups()
            return f'{float(value)} {unit}'
    return "" 
# Process each row in the test dataset and make predictions
predictions= []
for index, row in test_df.iterrows():
    image_url = row['image_link']
    entity_name = row['entity_name']
    
    # Download the image
    img= download_image(image_url)
    
    # Extract text from the image
    text= extract_text_from_image(img)
    
    # Extract the value and unit based on the entity name
    extracted_value = extract_value_and_unit(text, entity_name)
    
    
    # Append the prediction to the list
    predictions.append({'index': row['index'], 'prediction' :extracted_value })
    
# create the output Dataframe 
output_df = pd.DataFrame(predictions)

# save the output to csv file in the required format
output_file_path = 'd:/python projects/files/projectss/test_outs.csv'
output_df.to_csv(output_file_path, index= False)

print("Prediction complete. Output saved to 'test_outs.csv'.")