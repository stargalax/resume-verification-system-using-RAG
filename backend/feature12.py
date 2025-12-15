import PyPDF2
import re
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os
import json
import fitz
from typing import List, Dict, Any
from bs4 import BeautifulSoup

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------- PDF Extraction -------------------------
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

def split_into_lines(text):
    return [l.strip() for l in text.split("\n") if l.strip()]

# ------------------------- Vectorstore + RAG -------------------------
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain

# ------------------------- SECTION IDENTIFICATION (UPFRONT) -------------------------

def identify_all_sections(text, qa_chain):
    """
    STEP 1: Scan the entire resume and identify ALL relevant sections upfront.
    Returns a dict with section names and their content.
    """
    prompt = f"""
You are analyzing a resume. Your task is to IDENTIFY and EXTRACT specific sections.

Find these TWO sections:

1. ACHIEVEMENT SECTION - Look for headers like:
   - Achievements, Awards, Honors, Publications, Open Source Contributions, Competitions, Hackathons
   - Should contain: open source contributions, research papers, hackathon/competition wins
   - Should NOT contain: projects, work experience, internships

2. CERTIFICATION SECTION - Look for headers like:
   - Certifications, Certificates, Courses, Training, Professional Development, Online Courses, Licenses
   - Should contain: professional certifications, online courses, training programs

IMPORTANT: If you find a section, set "found" to true and include the FULL section content.

Return ONLY valid JSON in this exact format (no extra text):
{{
  "achievement_section": {{
    "found": true,
    "content": "full extracted section text here"
  }},
  "certification_section": {{
    "found": true,
    "content": "full extracted section text here"
  }}
}}

If a section is not found, use:
{{
  "found": false,
  "content": ""
}}

Resume text:
{text}
"""
    
    response = qa_chain.run(prompt)
    
    # Debug: print raw response
    print(f"\n[DEBUG] Raw LLM Response:\n{response}\n")
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        sections = json.loads(response_clean)
        
        # Validate and fix the structure
        if "achievement_section" not in sections:
            sections["achievement_section"] = {"found": False, "content": ""}
        if "certification_section" not in sections:
            sections["certification_section"] = {"found": False, "content": ""}
        
        # Auto-detect if content exists but found=false (LLM mistake)
        if sections["achievement_section"]["content"] and not sections["achievement_section"]["found"]:
            sections["achievement_section"]["found"] = True
            print("[DEBUG] Auto-corrected achievement_section found flag")
        
        if sections["certification_section"]["content"] and not sections["certification_section"]["found"]:
            sections["certification_section"]["found"] = True
            print("[DEBUG] Auto-corrected certification_section found flag")
        
        return sections
    except Exception as e:
        print(f"[ERROR] JSON parsing error: {e}")
        print(f"[ERROR] Raw response: {response}")
        return {
            "achievement_section": {"found": False, "content": ""},
            "certification_section": {"found": False, "content": ""}
        }

# ------------------------- FEATURE 1: ACHIEVEMENT VERIFICATION -------------------------

def extract_achievements_from_section(section_text, qa_chain):
    """
    Extract individual achievements from the identified section.
    """
    prompt = f"""
From this achievement section, extract each individual achievement.

For each achievement, extract:
- title: Brief title of the achievement
- type: "open_source" or "publication" or "hackathon"
- event_name: Name of the conference/hackathon/project
- level: "college" or "national" or "international" or "unknown"
- year: Year if mentioned
- details: Specific verifiable details (conference name, GitHub repo, hackathon organizer, rank obtained)

Return ONLY valid JSON in this exact format (no extra text, no markdown):
{{
  "achievements": [
    {{
      "title": "",
      "type": "",
      "event_name": "",
      "level": "",
      "year": "",
      "details": ""
    }}
  ]
}}

Achievement section text:
{section_text}
"""
    
    response = qa_chain.run(prompt)
    
    print(f"\n[DEBUG] Raw achievement extraction response:\n{response}\n")
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        # Remove any text before the first { and after the last }
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        return json.loads(response_clean)
    except Exception as e:
        print(f"[ERROR] JSON parsing error: {e}")
        print(f"[ERROR] Cleaned response: {response_clean}")
        return {"achievements": []}

def verify_achievement_online(achievement: Dict) -> Dict:
    """
    Verify each achievement by searching online.
    Returns verification result.
    """
    achievement_type = achievement.get('type', '')
    event_name = achievement.get('event_name', '')
    year = achievement.get('year', '')
    details = achievement.get('details', '')
    
    # Build search query based on type
    if achievement_type == 'publication':
        search_query = f'"{achievement["title"]}" {event_name} {year} publication paper'
    elif achievement_type == 'hackathon':
        search_query = f'{event_name} {year} hackathon winners results'
    elif achievement_type == 'open_source':
        search_query = f'{event_name} github open source'
    else:
        search_query = f'{event_name} {details} {year}'
    
    print(f"\n🔍 Searching: {search_query}")
    
    try:
        # Use DuckDuckGo HTML search
        encoded_query = requests.utils.quote(search_query)
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('a', class_='result__a')[:5]
            
            verification_info = {
                'verified': len(results) > 0,
                'search_query': search_query,
                'results_found': len(results),
                'top_results': []
            }
            
            for result in results[:3]:
                title = result.get_text().strip()
                href = result.get('href', '')
                verification_info['top_results'].append({
                    'title': title,
                    'url': href
                })
            
            return verification_info
        else:
            return {
                'verified': False,
                'search_query': search_query,
                'error': f'Search failed with status {response.status_code}'
            }
            
    except Exception as e:
        return {
            'verified': False,
            'search_query': search_query,
            'error': str(e)
        }

# ------------------------- FEATURE 2: CERTIFICATION WITH LINKS -------------------------

def extract_certifications_from_section(section_text, qa_chain):
    """
    Extract individual certifications from the identified section.
    """
    prompt = f"""
From this certification section, extract each individual certification or course.

For each, extract:
- name: Full name of the certification/course
- issuer: Organization that issued it (e.g., Coursera, Google, AWS, IBM, Oracle)
- year: Year completed if mentioned (leave empty if not mentioned)

Return ONLY valid JSON in this exact format (no extra text, no markdown, no explanation):
{{
  "certifications": [
    {{
      "name": "Python for Data Science",
      "issuer": "IBM",
      "year": ""
    }}
  ]
}}

Certification section text:
{section_text}
"""
    
    response = qa_chain.run(prompt)
    
    print(f"\n[DEBUG] Raw certification extraction response:\n{response}\n")
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        # Remove any text before the first { and after the last }
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        
        print(f"[DEBUG] Parsed {len(result.get('certifications', []))} certifications")
        for cert in result.get('certifications', []):
            print(f"   - {cert.get('name', 'N/A')} ({cert.get('issuer', 'N/A')})")
        
        return result
    except Exception as e:
        print(f"[ERROR] JSON parsing error: {e}")
        print(f"[ERROR] Cleaned response: {response_clean if 'response_clean' in locals() else response}")
        return {"certifications": []}

def get_certification_lines(certifications, all_lines):
    """
    Get lines from resume that mention certifications.
    """
    cert_lines = []
    for cert in certifications.get("certifications", []):
        name = cert.get("name", "")
        issuer = cert.get("issuer", "")
        
        for line in all_lines:
            # Match by name or issuer
            if (name and name.lower() in line.lower()) or (issuer and issuer.lower() in line.lower()):
                cert_lines.append(line.strip())
    
    return list(set(cert_lines))

def extract_hyperlinks_from_specific_text(pdf_path, target_lines):
    """
    Extract hyperlinks from the PDF that match any of the target lines.
    """
    doc = fitz.open(pdf_path)
    targeted_links = []
    target_lines_lower = [line.lower() for line in target_lines]

    for page_num in range(len(doc)):
        page = doc[page_num]
        for link in page.get_links():
            uri = link.get("uri", "")
            rect = link.get("from", None)
            if uri and rect:
                link_text = page.get_textbox(rect).strip()
                if link_text:
                    # Check if link text matches any certification line
                    is_match = any(
                        target_line in link_text.lower() or link_text.lower() in target_line
                        for target_line in target_lines_lower
                    )
                    if is_match and "mailto:" not in uri:
                        targeted_links.append({
                            "text": link_text,
                            "hyperlink": uri,
                            "page": page_num + 1
                        })
    doc.close()
    return targeted_links

def match_certifications_with_links(certifications, extracted_links):
    """
    Match certification names with extracted links.
    """
    certs_with_links = []
    certs_without_links = []
    
    for cert in certifications.get("certifications", []):
        cert_name = cert.get("name", "").lower()
        cert_issuer = cert.get("issuer", "").lower()
        
        # Try to find matching link
        matched_link = None
        for link in extracted_links:
            link_text_lower = link['text'].lower()
            # Check if cert name or issuer appears in link text
            if cert_name in link_text_lower or link_text_lower in cert_name or \
               (cert_issuer and cert_issuer in link_text_lower):
                matched_link = link
                break
        
        if matched_link:
            certs_with_links.append({
                "name": cert["name"],
                "issuer": cert.get("issuer", ""),
                "year": cert.get("year", ""),
                "url": matched_link["hyperlink"]
            })
        else:
            certs_without_links.append({
                "name": cert["name"],
                "issuer": cert.get("issuer", ""),
                "year": cert.get("year", "")
            })
    
    return certs_with_links, certs_without_links

# ------------------------- MAIN -------------------------

def main():
    resume_path = "reseumee.pdf"
    
    print("="*80)
    print("RESUME VERIFICATION SYSTEM")
    print("="*80)
    
    # Extract text
    text = extract_text_from_pdf(resume_path)
    lines = split_into_lines(text)
    
    # Create RAG system
    vectorstore = create_vector_store(text)
    qa_chain = create_qa_chain(vectorstore)
    
    # ============= UPFRONT: IDENTIFY ALL SECTIONS =============
    print("\n🔍 SCANNING RESUME TO IDENTIFY SECTIONS...")
    print("-"*80)
    
    sections = identify_all_sections(text, qa_chain)
    
    # Report what was found
    achievement_found = sections['achievement_section']['found']
    certification_found = sections['certification_section']['found']
    
    print(f"\n📋 Achievement Section: {'✅ FOUND' if achievement_found else '❌ NOT FOUND'}")
    if achievement_found:
        preview = sections['achievement_section']['content'][:200]
        print(f"   Preview: {preview}...")
    
    print(f"\n📜 Certification Section: {'✅ FOUND' if certification_found else '❌ NOT FOUND'}")
    if certification_found:
        preview = sections['certification_section']['content'][:200]
        print(f"   Preview: {preview}...")
    
    # ============= FEATURE 1: ACHIEVEMENTS =============
    print("\n\n" + "="*80)
    print("FEATURE 1: EXTRACTING AND VERIFYING ACHIEVEMENTS")
    print("="*80)
    
    if not achievement_found:
        print("\n⚠️  SKIPPING: No achievement section found in resume")
        verified_achievements = []
    else:
        achievement_section = sections['achievement_section']['content']
        
        # Extract achievements from section
        print("\n🔍 Extracting individual achievements from section...")
        achievements_data = extract_achievements_from_section(achievement_section, qa_chain)
        
        if not achievements_data.get("achievements"):
            print("⚠️  No achievements extracted from section")
            verified_achievements = []
        else:
            verified_achievements = []
            for idx, achievement in enumerate(achievements_data.get("achievements", []), 1):
                print(f"\n[{idx}] {achievement['title']}")
                print(f"    Type: {achievement['type']}")
                print(f"    Event: {achievement.get('event_name', 'N/A')}")
                print(f"    Level: {achievement.get('level', 'N/A')}")
                
                # Verify online
                print(f"    🔍 Verifying online...")
                verification = verify_achievement_online(achievement)
                achievement['verification'] = verification
                
                if verification['verified']:
                    print(f"    ✅ VERIFIED - Found {verification['results_found']} results")
                    if verification['top_results']:
                        print(f"    Top result: {verification['top_results'][0]['title']}")
                else:
                    print(f"    ❌ NOT VERIFIED - {verification.get('error', 'No results found')}")
                
                verified_achievements.append(achievement)
    
    # ============= FEATURE 2: CERTIFICATIONS =============
    print("\n\n" + "="*80)
    print("FEATURE 2: EXTRACTING CERTIFICATIONS WITH LINKS")
    print("="*80)
    
    if not certification_found:
        print("\n⚠️  SKIPPING: No certification section found in resume")
        certs_with_links = []
        certs_without_links = []
    else:
        certification_section = sections['certification_section']['content']
        
        # Extract certifications from section
        print("\n🔍 Extracting individual certifications from section...")
        certifications_data = extract_certifications_from_section(certification_section, qa_chain)
        
        if not certifications_data.get('certifications'):
            print("⚠️  No certifications extracted from section")
            certs_with_links = []
            certs_without_links = []
        else:
            print(f"Found {len(certifications_data.get('certifications', []))} certifications")
            
            # Get certification lines from original text
            cert_lines = get_certification_lines(certifications_data, lines)
            print(f"Identified {len(cert_lines)} certification-related lines in resume")
            
            # Extract links from PDF
            print(f"🔍 Extracting hyperlinks from PDF...")
            extracted_links = extract_hyperlinks_from_specific_text(resume_path, cert_lines)
            print(f"Extracted {len(extracted_links)} hyperlinks from certification sections")
            
            # Match certs with links
            print(f"🔍 Matching certifications with extracted links...")
            certs_with_links, certs_without_links = match_certifications_with_links(
                certifications_data, extracted_links
            )
    
    print(f"\n✅ {len(certs_with_links)} certifications WITH links")
    print(f"❌ {len(certs_without_links)} certifications WITHOUT links")
    
    print("\n--- Certifications with Links ---")
    for cert in certs_with_links:
        print(f"  • {cert['name']}")
        print(f"    Issuer: {cert['issuer']}")
        print(f"    Link: {cert['url']}\n")
    
    print("--- Certifications without Links ---")
    for cert in certs_without_links:
        print(f"  • {cert['name']} ({cert['issuer']})")
    
    # ============= SAVE RESULTS =============
    output = {
        "achievements": verified_achievements,
        "certifications": {
            "with_links": certs_with_links,
            "without_links": certs_without_links
        }
    }
    
    with open("verification_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ Results saved to verification_results.json")
    print("="*80)

if __name__ == "__main__":
    main()