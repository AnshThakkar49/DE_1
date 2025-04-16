# report_analysis.py

import pandas as pd
import spacy
import pdfplumber
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

reference_ranges = {
    "Hemoglobin": (13.8, 17.2),
    "WBC": (4000, 11000),
    "Platelets": (150000, 450000),
    "RBC": (4.7, 6.1),
    "Hematocrit": (40.7, 50.3),
    "MCV": (80, 100),
    "MCH": (27, 33),
    "MCHC": (32, 36),
    "RDW": (11.5, 14.5),
    "Glucose": (70, 99),
    "Cholesterol": (0, 200),
    "Triglycerides": (0, 150),
    "Creatinine": (0.7, 1.3),
    "Urea": (7, 20),
}

def load_report(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                all_text += page.extract_text() + "\n"
        lines = all_text.strip().splitlines()
        data = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                parameter = " ".join(parts[:-1])
                value = parts[-1]
                data.append([parameter, value])
        return pd.DataFrame(data, columns=["parameter", "value"])
    else:
        raise ValueError("Unsupported file format")

def analyze_blood_report(file_path):
    df = load_report(file_path)
    results = []
    for _, row in df.iterrows():
        parameter = row["parameter"]
        try:
            value = float(row["value"].replace(",", ""))
        except ValueError:
            results.append(f"{parameter}: Invalid value detected.")
            continue
        if parameter in reference_ranges:
            lower, upper = reference_ranges[parameter]
            if value < lower:
                results.append(f"{parameter}: Low ({value}). Normal: {lower}-{upper}")
            elif value > upper:
                results.append(f"{parameter}: High ({value}). Normal: {lower}-{upper}")
            else:
                results.append(f"{parameter}: Normal ({value})")
        else:
            results.append(f"{parameter}: No reference range available.")
    return results

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}
