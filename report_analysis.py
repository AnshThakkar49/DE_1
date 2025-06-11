import pandas as pd
import spacy
import pdfplumber
import os
import re
from datetime import datetime
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
            match = re.search(r"([a-zA-Z ]+)\s+([\d.,]+)", line)
            if match:
                parameter = match.group(1).strip()
                value = match.group(2).replace(",", "")
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
            #results.append(f"{parameter}: Invalid value detected.")
            continue
        if parameter in reference_ranges:
            lower, upper = reference_ranges[parameter]
            if value < lower:
                results.append(f"{parameter}: Low ({value}). Normal: {lower}-{upper}")
            elif value > upper:
                results.append(f"{parameter}: High ({value}). Normal: {lower}-{upper}")
            else:
                results.append(f"{parameter}: Normal ({value})")
    return results

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}

def generate_html_report(results, output_path="blood_report_summary.html"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""
    <html>
    <head>
        <title>Blood Report Summary</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            h1 {{ color: #333; }}
            .low {{ color: #d9534f; }}
            .high {{ color: #f0ad4e; }}
            .normal {{ color: #5cb85c; }}
            .unknown {{ color: #999; }}
        </style>
    </head>
    <body>
        <h1>Blood Report Analysis Summary</h1>
        <p><strong>Date:</strong> {now}</p>
        <ul>
    """

    for result in results:
        if "Low" in result:
            css_class = "low"
        elif "High" in result:
            css_class = "high"
        elif "Normal" in result:
            css_class = "normal"
        else:
            css_class = "unknown"
        html += f"<li class='{css_class}'>{result}</li>\n"

    html += """
        </ul>
        <p><em>This report is auto-generated for preliminary insights and is not a substitute for medical advice.</em></p>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ HTML report saved to: {output_path}")
