# ============================================================
# BUILD.PY – ONE-TIME SETUP FOR CONNOISSEUR COMPANION
# ============================================================
# This script downloads data, processes it with LLM,
# and builds the vector database. Run it once before using app.py.
# ============================================================

import os, json, requests, zipfile
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================================
# CONFIGURATION – API KEY & PATHS
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY environment variable")
client = OpenAI(api_key=OPENAI_API_KEY)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# HELPER FUNCTIONS (LLM with retry, JSON cleaning)
# ============================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def llm(system, user, temp=0.3):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temp
    )
    return resp.choices[0].message.content

def clean_json(text):
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()

# ============================================================
# PHASE 1: DOWNLOAD RAW DATA (SKIP IF EXISTS)
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Checking/Downloading raw data files")
print("="*60)

urls = {
    "California-Culinary-Map.txt": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1r_mM6ZPYNxcFv65QkzubA/California-Culinary-Map.txt",
    "Recipes.json": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hpTjb6liKBLVHQK0UgMi5A/Recipes.json",
    "Synthetic-User-Reviews.json": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fQUs9wQ6aB6ts6fmkD2V2w/Synthetic-User-Reviews.json",
    "synthetic-recipe-images.zip": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"
}
for name, url in urls.items():
    dest = DATA_DIR / name
    if dest.exists():
        print(f"✅ Already have {name}")
    else:
        print(f"⬇️ Downloading {name}...")
        dest.write_bytes(requests.get(url).content)

# Extract images if not already done
img_dir = DATA_DIR / "recipe_images"
if not img_dir.exists():
    print("📦 Extracting images...")
    with zipfile.ZipFile(DATA_DIR / "synthetic-recipe-images.zip", 'r') as zf:
        zf.extractall(img_dir)
else:
    print("✅ Images already extracted")

# ============================================================
# PHASE 2: BUILD STRUCTURED RESTAURANT JSON (FIRST 20)
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Structuring restaurant descriptions into JSON")
print("="*60)

rest_json = OUTPUT_DIR / "restaurants.json"
if rest_json.exists():
    print(f"✅ {rest_json} already exists – skipping (delete it to rebuild)")
else:
    print("Processing restaurant data (LLM calls – one-time, cost ~₹0.02)...")
    with open(DATA_DIR / "California-Culinary-Map.txt") as f:
        raw = f.read()
    paragraphs = raw.split("\n\n")[1:21]          # First 20 restaurants
    system = "Extract restaurant data to JSON. Fields: name, location, food_style, rating, price_range (number of $), vibe. Return ONLY JSON."
    structured = []
    for i, para in enumerate(paragraphs, 1):
        try:
            print(f"  {i}/20: processing...")
            resp = llm(system, f"Convert to JSON: {para}")
            structured.append(json.loads(clean_json(resp)))
        except Exception as e:
            print(f"  Error: {e}")
            structured.append({"name": "Unknown", "error": str(e)})
    with open(rest_json, 'w') as f:
        json.dump(structured, f, indent=2)
    print(f"✅ Saved {len(structured)} restaurants to {rest_json}")

# ============================================================
# PHASE 3: BUILD CHROMA VECTOR DATABASE
# ============================================================
print("\n" + "="*60)
print("PHASE 3: Building Chroma vector database")
print("="*60)

vecdb_dir = OUTPUT_DIR / "chroma_db"
if vecdb_dir.exists():
    print(f"✅ Vector DB already exists at {vecdb_dir} – skipping")
else:
    print("Generating embeddings and storing in Chroma...")
    with open(rest_json) as f:
        restaurants = json.load(f)
    text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    docs = []
    for i, r in enumerate(restaurants):
        content = f"{r.get('name','')} {r.get('food_style','')} {r.get('location','')} {r.get('vibe','')}"
        docs.append(Document(
            page_content=content,
            metadata={
                "id": i,
                "name": r.get('name'),
                "cuisine": r.get('food_style'),
                "location": r.get('location')
            }
        ))
    Chroma.from_documents(docs, text_embedder, persist_directory=str(vecdb_dir))
    print(f"✅ Vector DB built with {len(docs)} restaurants")

# ============================================================
# DONE
# ============================================================
print("\n" + "="*60)
print("✅ BUILD COMPLETE")
print("="*60)
print("You can now run: python app.py")
