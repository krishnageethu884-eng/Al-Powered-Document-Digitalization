import cv2
import pytesseract
import pandas as pd
import os
from PIL import Image
import re
import numpy as np
from pdf2image import convert_from_bytes
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\krish\Downloads\datascience\deep_learning_prjct\OCR\Tesseract-OCR\tesseract.exe"

def preprocess_image(img):
    """
    Preprocesses the image (numpy array).
    Simplified to just resize and grayscale to avoid over-processing.
    """
    try:
        if img is None:
            return None
        
        # Helper to resize only if image is small/low-res
        h, w = img.shape[:2]
        if w < 1000:
            scale = 2
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Skip aggressive thresholding and blurring for now
        # simple thresholding to ensure high contrast without noise removal artifacts
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return gray
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def parse_fields(text):
    data = {}
    
    
    def get_value(labels, stop_words, text):
        label_part = "|".join(labels)
        pattern = r"(?:" + label_part + r")\s*[:\.\-]?\s*([^\n]*)(?:\n\s*([^\n]+))?"
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            
            val1 = match.group(1).strip() if match.group(1) else ""
            val2 = match.group(2).strip() if match.group(2) else ""
            
           
            candidate = val1
            
        
            is_valid = True
            if len(candidate) < 2: is_valid = False
            
            if any(sw.lower() in candidate.lower() for sw in stop_words):
                
                 if len(candidate) < 15: 
                      is_valid = False
            
            if not is_valid and val2:
                candidate = val2
            
            
            val = candidate
            
            for sw in stop_words:
                sw_idx = val.lower().find(sw.lower())
                if sw_idx != -1:
                    val = val[:sw_idx].strip()
            
            if "name of" in val.lower() and len(val) < 10:
                return ""
                
            return val
        return ""

    def clean_name(val):
        if not val: return ""
        
      
        match_lower = re.search(r'[a-z]', val)
        if match_lower:
            val = val[:match_lower.start()]
            
        
        val = re.sub(r'[^a-zA-Z\s\.]', '', val)
        val = re.sub(r'\s+', ' ', val).strip()
        
        val = val.lstrip('.')
        return val

    def clean_dob(val):
        if not val: return ""
        
        match = re.search(r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})', val)
        if match:
            d = match.group(1).replace('-', '/')
            return d
        return ""

    def clean_sex(val):
        if not val: return ""
        val_lower = val.lower()
        if "female" in val_lower:
            return "Female"
        if "male" in val_lower:
            return "Male"
        return ""

    
    reg_match = re.search(r"(?:Register|Reg)\.?\s*(?:Number|No)\.?\s*[:\.\-]?\s*(\d{3,})", text, re.IGNORECASE)
    if reg_match:
        data['register_number'] = reg_match.group(1)
    else:
        adm_match = re.search(r"Admission\s*No\.?\s*[:\.\-]?\s*(\d+)", text, re.IGNORECASE)
        data['register_number'] = adm_match.group(1) if adm_match else ""

    
    raw_name = get_value(
        ["Name of Candidate", "Name of Student", "Name", "Nume", "Nome", "Candidate Name"], 
        ["Date", "Sex", "Male", "Female", "Name of Father", "Father", "Religion", "Caste"], 
        text
    )
    
    if any(x in raw_name.lower() for x in ["school", "certified", "guardian", "headmaster"]):
        data['name'] = ""
    else:
        data['name'] = clean_name(raw_name)

    
    sex_text = get_value(["Sex", "Gender", "Male", "Female"], ["Date", "Nationality", "Religion", "Caste"], text)
    
    if not sex_text:
        if re.search(r"\bFemale\b", text, re.IGNORECASE):
            data['sex'] = "Female"
        elif re.search(r"\bMale\b", text, re.IGNORECASE):
            data['sex'] = "Male"
        else:
            data['sex'] = ""
    else:
        data['sex'] = clean_sex(sex_text)
    
    
    dob_match = re.search(r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})", text)
    if dob_match:
        data['dob'] = dob_match.group(1).replace('.', '/').replace('-', '/')
    else:
        raw_dob = get_value(["Date of Birth", "DOB", "Dale of Birth", "Date of Bith"], ["Male", "Female", "Sex", "Father"], text)
        data['dob'] = clean_dob(raw_dob)

    
    raw_father = get_value(
        ["Name of Father", "Father Name", "Father's Name", "Father"], 
        ["Mother", "Guardian", "Address", "Occupation", "Mother's"], 
        text
    )
    data['father_name'] = clean_name(raw_father)

    
    raw_mother = get_value(
        ["Name of Mother", "Mother Name", "Mother's Name", "Mother"], 
        ["Guardian", "Address", "Occupation", "Place"], 
        text
    )
    data['mother_name'] = clean_name(raw_mother)
    
    def clean_address(val):
        if not val: return ""
        
        
        match_lower = re.search(r'[a-z]', val)
        if match_lower:
             val = val[:match_lower.start()]
             
        
        val = re.sub(r'\d+', '', val)
        
        val = re.sub(r'[^a-zA-Z\s]', ' ', val)
        
        val = re.sub(r'\s+', ' ', val).strip()
        return val

   
    address_match = re.search(r"(?:Address|Permanent\s*Address)\s*[:\.\-]?\s*([\s\S]+?)(?=\n\d|Identification|Marks|Headmaster)", text, re.IGNORECASE)
    if address_match:
        raw_addr = address_match.group(1).strip().replace('\n', ' ')
        data['address'] = clean_address(raw_addr)
    else:
        data['address'] = ""

    return data



def process_single_image(img, filename_label):
   
    preprocessed_img = preprocess_image(img)
    if preprocessed_img is not None:
        text = pytesseract.image_to_string(Image.fromarray(preprocessed_img))
        print(f"\n--- RAW OCR OUTPUT ({filename_label}) ---\n{text}\n----------------------------------\n")
        extracted_data = parse_fields(text)
        extracted_data['filename'] = filename_label
        return extracted_data
    return None

def process_file_data(file_bytes, filename):
    """
    Processes a file (Image or PDF) from bytes and returns a list of data dicts.
    Returns a LIST because a PDF might have multiple pages (though usually we expect 1 student per file).
    """
    results = []
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext == '.pdf':
            # Convert PDF to images
            images = convert_from_bytes(file_bytes)
            for i, page_img in enumerate(images):
                # Convert PIL image to numpy array (OpenCV format)
                page_arr = np.array(page_img)
                # Convert RGB to BGR
                page_arr = cv2.cvtColor(page_arr, cv2.COLOR_RGB2BGR)
                
                label = f"{filename}_page_{i+1}"
                data = process_single_image(page_arr, label)
                if data:
                    results.append(data)
        else:
            # Assuming Image
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            data = process_single_image(img, filename)
            if data:
                results.append(data)
                
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        
    return results

def process_image_from_memory(file_bytes, filename):
   
    res = process_file_data(file_bytes, filename)
    return res[0] if res else None


