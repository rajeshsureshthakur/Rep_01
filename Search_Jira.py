# defect_assistant.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser
import os

# Read saved query
if not os.path.exists("query.txt"):
    print("No query found.")
    exit()

with open("query.txt", "r", encoding="utf-8") as f:
    user_query = f.read().strip()

# Load defect data
try:
    if pd.__version__ >= '1.3.0':
        df = pd.read_csv("defects.csv", on_bad_lines='skip')
    else:
        df = pd.read_csv("defects.csv", error_bad_lines=False)
except Exception as e:
    print(f"CSV Load Error: {e}")
    exit()

df = df.dropna(subset=["Description"])
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

# Find similar issues
def find_similar_issues(query, top_n=3):
    user_tfidf = vectorizer.transform([query])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    sorted_indices = cosine_sim.argsort()[::-1]

    html = """
    <html><head><title>Results</title><style>
    body { font-family: Arial; padding: 20px; background: #f9f9f9; }
    table { width: 100%; border-collapse: collapse; background: white; }
    th, td { border: 1px solid #ddd; padding: 8px; }
    th { background: #4CAF50; color: white; }
    tr:nth-child(even) { background: #f2f2f2; }
    </style></head><body><h2>🔎 Similar Issues</h2><table>
    <tr><th>#</th><th>Issue Key</th><th>Summary</th><th>Status</th><th>Description</th><th>Resolution</th></tr>
    """

    seen = set()
    count = 0
    for idx in sorted_indices:
        desc = df.iloc[idx]['Description'].strip().lower()
        if desc in seen or cosine_sim[idx] < 0.01:
            continue
        seen.add(desc)
        row = df.iloc[idx]
        html += f"<tr><td>{count+1}</td><td>{row.get('Issue key', '')}</td><td>{row.get('Summary', '')}</td><td>{row.get('Status', '')}</td><td>{row.get('Description', '')}</td><td>{row.get('Comment', '')}</td></tr>"
        count += 1
        if count >= top_n:
            break

    html += "</table></body></html>"
    return html

html_output = find_similar_issues(user_query)

# Save & open
with open("defect_results.html", "w", encoding="utf-8") as f:
    f.write(html_output)
webbrowser.open('file://' + os.path.realpath("defect_results.html"))
