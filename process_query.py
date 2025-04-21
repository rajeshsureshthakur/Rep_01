import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cgi
import os

# Load the historical defect data with error handling for CSV parsing issues
def load_data():
    try:
        if pd.__version__ >= '1.3.0':
            df = pd.read_csv("defects.csv", on_bad_lines='skip')  # For newer pandas versions
        else:
            df = pd.read_csv("defects.csv", error_bad_lines=False)  # For older pandas versions
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Vectorize the issue descriptions
def prepare_vectorizer(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Description'])
    return vectorizer, tfidf_matrix

# Function to find similar issues with duplicate filtering
def find_similar_issues(user_input, df, vectorizer, tfidf_matrix, top_n=3, similarity_threshold=0.95):
    user_tfidf = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Get all indices sorted by similarity
    sorted_indices = cosine_sim.argsort()[::-1]
    
    # Filter to keep only unique content
    unique_issues = []
    seen_descriptions = set()
    
    for idx in sorted_indices:
        description = df.iloc[idx]['Description']
        simple_desc = description.lower().strip()
        
        if simple_desc in seen_descriptions or cosine_sim[idx] < 0.01:
            continue
        
        too_similar = False
        for existing_idx in unique_issues:
            existing_desc = df.iloc[existing_idx]['Description']
            if cosine_similarity(
                vectorizer.transform([description]), 
                vectorizer.transform([existing_desc])
            )[0][0] > similarity_threshold:
                too_similar = True
                break
                
        if not too_similar:
            unique_issues.append(idx)
            seen_descriptions.add(simple_desc)
            
        if len(unique_issues) >= top_n:
            break
    
    return unique_issues, cosine_sim

# Handle the form input
form = cgi.FieldStorage()
user_query = form.getvalue("user_query")

if user_query:
    df = load_data()
    if df is None:
        print("Error loading data. Please check your CSV file.")
    else:
        vectorizer, tfidf_matrix = prepare_vectorizer(df)
        unique_issues, cosine_sim = find_similar_issues(user_query, df, vectorizer, tfidf_matrix)

        # Prepare the results for HTML output
        result_data = []
        for idx in unique_issues:
            issue_data = {
                "Issue Key": df.iloc[idx].get('Issue key', 'N/A'),
                "Summary": df.iloc[idx].get('Summary', 'N/A'),
                "Status": df.iloc[idx].get('Status', 'N/A'),
                "Fix Version": df.iloc[idx].get('Fix Version', 'N/A'),
                "Severity": df.iloc[idx].get('Severity', 'N/A'),
                "Description": df.iloc[idx]['Description'],
                "Resolution": df.iloc[idx].get('Comment', 'N/A'),
                "Similarity Score": f"{cosine_sim[idx]:.2f}"
            }
            result_data.append(issue_data)

        # Generate HTML result
        print("Content-type: text/html\n")
        print("""
        <html>
        <head>
            <title>Defect Assistant - Similar Issues</title>
            <style>
                body {font-family: Arial, sans-serif; padding: 20px;}
                table {width: 100%; border-collapse: collapse;}
                th, td {padding: 10px; text-align: left; border: 1px solid #ddd;}
                th {background-color: #f2f2f2;}
            </style>
        </head>
        <body>
            <h2>🔍 Top similar past issues:</h2>
            <table>
                <tr>
                    <th>Issue Key</th>
                    <th>Summary</th>
                    <th>Status</th>
                    <th>Fix Version</th>
                    <th>Severity</th>
                    <th>Description</th>
                    <th>Resolution</th>
                    <th>Similarity Score</th>
                </tr>
        """)

        for issue in result_data:
            print(f"""
            <tr>
                <td>{issue['Issue Key']}</td>
                <td>{issue['Summary']}</td>
                <td>{issue['Status']}</td>
                <td>{issue['Fix Version']}</td>
                <td>{issue['Severity']}</td>
                <td>{issue['Description']}</td>
                <td>{issue['Resolution']}</td>
                <td>{issue['Similarity Score']}</td>
            </tr>
            """)

        print("</table></body></html>")
