import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import PyPDF2
import os

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Sample training data (replace with real dataset)
data = {
    "resume_text": [
        "Experienced software engineer with expertise in Python and machine learning.",
        "Marketing specialist with a background in social media and branding.",
        "Data scientist skilled in deep learning and artificial intelligence.",
        "Mechanical engineer with CAD design and simulation experience."
    ],
    "category": ["Software Engineer", "Marketing", "Data Scientist", "Mechanical Engineer"]
}
df = pd.DataFrame(data)

# Vectorizing the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["resume_text"])
y = df["category"]

# Train SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X, y)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def classify_resume(resume_text):
    processed_text = vectorizer.transform([resume_text])
    prediction = classifier.predict(processed_text)
    return prediction[0]

def process_resumes(folder_path):
    results = {}
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(folder_path, file))
            category = classify_resume(text)
            results[file] = category
            print(f"Resume: {file} -> Classified as: {category}")
    return results

# Example usage
if __name__ == "__main__":
    resume_folder = "resumes/"  # Replace with actual path
    classifications = process_resumes(resume_folder)
    print("Resume Screening Completed")
