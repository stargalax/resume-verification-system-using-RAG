

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
import tempfile
import shutil
from pathlib import Path

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Resume Verification API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================= HELPER FUNCTIONS =========================

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

def split_into_lines(text):
    """Split text into lines"""
    return [l.strip() for l in text.split("\n") if l.strip()]

def create_vector_store(text):
    """Create FAISS vector store from text"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    """Create QA chain with Groq LLM"""
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

# ========================= SECTION IDENTIFICATION =========================

def identify_all_sections(text, qa_chain):
    """Identify achievement and certification sections"""
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
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        sections = json.loads(response_clean)
        
        if "achievement_section" not in sections:
            sections["achievement_section"] = {"found": False, "content": ""}
        if "certification_section" not in sections:
            sections["certification_section"] = {"found": False, "content": ""}
        
        if sections["achievement_section"]["content"] and not sections["achievement_section"]["found"]:
            sections["achievement_section"]["found"] = True
        
        if sections["certification_section"]["content"] and not sections["certification_section"]["found"]:
            sections["certification_section"]["found"] = True
        
        return sections
    except Exception as e:
        print(f"[ERROR] JSON parsing error: {e}")
        return {
            "achievement_section": {"found": False, "content": ""},
            "certification_section": {"found": False, "content": ""}
        }

# ========================= ACHIEVEMENT VERIFICATION =========================

def extract_achievements_from_section(section_text, qa_chain):
    """Extract individual achievements"""
    prompt = f"""
From this achievement section, extract each individual achievement.

For each achievement, extract:
- title: Brief title of the achievement
- type: "open_source" or "publication" or "hackathon"
- event_name: Name of the conference/hackathon/project
- level: "college" or "national" or "international" or "unknown"
- year: Year if mentioned
- details: Specific verifiable details

Return ONLY valid JSON in this exact format (no extra text):
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
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        return json.loads(response_clean)
    except Exception as e:
        print(f"[ERROR] JSON parsing error: {e}")
        return {"achievements": []}

def verify_achievement_online(achievement: Dict) -> Dict:
    """Verify achievement by searching online"""
    achievement_type = achievement.get('type', '')
    event_name = achievement.get('event_name', '')
    year = achievement.get('year', '')
    details = achievement.get('details', '')
    
    if achievement_type == 'publication':
        search_query = f'"{achievement["title"]}" {event_name} {year} publication paper'
    elif achievement_type == 'hackathon':
        search_query = f'{event_name} {year} hackathon winners results'
    elif achievement_type == 'open_source':
        search_query = f'{event_name} github open source'
    else:
        search_query = f'{event_name} {details} {year}'
    
    try:
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

# ========================= CERTIFICATION EXTRACTION =========================

def extract_certifications_from_section(section_text, qa_chain):
    """Extract individual certifications"""
    prompt = f"""
From this certification section, extract each individual certification or course.

For each, extract:
- name: Full name of the certification/course
- issuer: Organization that issued it
- year: Year completed if mentioned (leave empty if not mentioned)

Return ONLY valid JSON in this exact format (no extra text):
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
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        return json.loads(response_clean)
    except Exception as e:
        print(f"[ERROR] JSON parsing error: {e}")
        return {"certifications": []}

def get_certification_lines(certifications, all_lines):
    """Get lines from resume that mention certifications"""
    cert_lines = []
    for cert in certifications.get("certifications", []):
        name = cert.get("name", "")
        issuer = cert.get("issuer", "")
        
        for line in all_lines:
            if (name and name.lower() in line.lower()) or (issuer and issuer.lower() in line.lower()):
                cert_lines.append(line.strip())
    
    return list(set(cert_lines))

def extract_hyperlinks_from_specific_text(pdf_path, target_lines):
    """Extract hyperlinks from PDF"""
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
    """Match certifications with their verification links"""
    certs_with_links = []
    certs_without_links = []
    
    for cert in certifications.get("certifications", []):
        cert_name = cert.get("name", "").lower()
        cert_issuer = cert.get("issuer", "").lower()
        
        matched_link = None
        for link in extracted_links:
            link_text_lower = link['text'].lower()
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

# ========================= PROJECT EXTRACTION =========================

def identify_project_section(text, qa_chain):
    """Identify project section"""
    prompt = f"""
You are analyzing a resume. Your task is to IDENTIFY and EXTRACT the complete section that contains projects.

Look for sections with headers like:
- Projects, Personal Projects, Academic Projects, Work Projects, Portfolio

Return ONLY valid JSON in this exact format (no extra text):
{{
  "project_section": {{
    "found": true,
    "content": "full extracted section text here"
  }}
}}

Resume text:
{text}
"""
    
    response = qa_chain.run(prompt)
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        
        if result["project_section"]["content"] and not result["project_section"]["found"]:
            result["project_section"]["found"] = True
        
        return result
    except Exception as e:
        return {"project_section": {"found": False, "content": ""}}

def extract_top_5_projects(section_text, qa_chain):
    """Extract top 5 projects"""
    prompt = f"""
From this project section, extract the FIRST 5 PROJECTS in order of appearance.

For each project, extract:
- title: Project name/title
- description: Brief description
- technologies: List of technologies used
- domain: Domain/category

Return ONLY valid JSON in this exact format:
{{
  "projects": [
    {{
      "title": "Project Name",
      "description": "What the project does",
      "technologies": ["Python", "FastAPI"],
      "domain": "Machine Learning"
    }}
  ]
}}

Project section text:
{section_text}
"""
    
    response = qa_chain.run(prompt)
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        projects = result.get("projects", [])[:5]
        return {"projects": projects}
    except Exception as e:
        return {"projects": []}

# ========================= MCQ GENERATION =========================

def generate_role_based_mcqs(role: str, qa_chain) -> List[Dict]:
    """Generate 2 role-based MCQs"""
    prompt = f"""
Generate 2 medium-difficulty MCQ questions for a {role} position.

Requirements:
- Test core concepts
- Medium difficulty
- 4 options each
- Only 1 correct answer

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "Question text?",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "B",
      "explanation": "Explanation"
    }}
  ]
}}
"""
    
    response = qa_chain.run(prompt)
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        return result.get("questions", [])[:2]
    except Exception as e:
        return []

def generate_project_specific_mcqs(projects: List[Dict], qa_chain) -> List[Dict]:
    """Generate 3 project-specific MCQs"""
    all_techs = []
    project_summaries = []
    
    for proj in projects:
        all_techs.extend(proj.get("technologies", []))
        project_summaries.append(f"{proj['title']}: {proj['description']}")
    
    tech_summary = ", ".join(list(set(all_techs)))
    projects_text = "\n".join(project_summaries)
    
    prompt = f"""
Generate 3 medium-difficulty MCQ questions based on these projects:

Projects:
{projects_text}

Technologies: {tech_summary}

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "Question?",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "B",
      "explanation": "Explanation",
      "related_project": "Project title"
    }}
  ]
}}
"""
    
    response = qa_chain.run(prompt)
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        return result.get("questions", [])[:3]
    except Exception as e:
        return []

def generate_system_design_mcq(projects: List[Dict], qa_chain) -> Dict:
    """Generate 1 system design MCQ"""
    if not projects:
        return {}
    
    selected_project = projects[0]
    
    prompt = f"""
Generate 1 system design MCQ based on this project:

Project: {selected_project['title']}
Description: {selected_project['description']}

Return ONLY valid JSON:
{{
  "question": "Question?",
  "options": ["A", "B", "C", "D"],
  "correct_answer": "B",
  "explanation": "Explanation",
  "related_project": "{selected_project['title']}"
}}
"""
    
    response = qa_chain.run(prompt)
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        return json.loads(response_clean)
    except Exception as e:
        return {}

# ========================= API ENDPOINTS =========================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Resume Verification API"}

@app.post("/api/verify-resume")
async def verify_resume(
    file: UploadFile = File(...),
    role: str = Form("Python Developer")
):
    """
    Main endpoint to process resume and return all verification results
    """
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract text
        text = extract_text_from_pdf(temp_file_path)
        lines = split_into_lines(text)
        
        # Create RAG system
        vectorstore = create_vector_store(text)
        qa_chain = create_qa_chain(vectorstore)
        
        # Identify sections
        sections = identify_all_sections(text, qa_chain)
        
        # ============= FEATURE 1: ACHIEVEMENTS =============
        achievements_result = []
        if sections['achievement_section']['found']:
            achievement_section = sections['achievement_section']['content']
            achievements_data = extract_achievements_from_section(achievement_section, qa_chain)
            
            for achievement in achievements_data.get("achievements", []):
                verification = verify_achievement_online(achievement)
                achievement['verification'] = verification
                achievements_result.append(achievement)
        
        # ============= FEATURE 2: CERTIFICATIONS =============
        certs_with_links = []
        certs_without_links = []
        
        if sections['certification_section']['found']:
            certification_section = sections['certification_section']['content']
            certifications_data = extract_certifications_from_section(certification_section, qa_chain)
            
            if certifications_data.get('certifications'):
                cert_lines = get_certification_lines(certifications_data, lines)
                extracted_links = extract_hyperlinks_from_specific_text(temp_file_path, cert_lines)
                certs_with_links, certs_without_links = match_certifications_with_links(
                    certifications_data, extracted_links
                )
        
        # ============= FEATURE 3: MCQs =============
        mcq_result = {
            "success": False,
            "questions": []
        }
        
        project_section_data = identify_project_section(text, qa_chain)
        
        if project_section_data["project_section"]["found"]:
            project_section = project_section_data["project_section"]["content"]
            projects_data = extract_top_5_projects(project_section, qa_chain)
            projects = projects_data.get("projects", [])
            
            if projects:
                all_questions = []
                
                # Generate MCQs
                role_mcqs = generate_role_based_mcqs(role, qa_chain)
                for q in role_mcqs:
                    q['category'] = 'role_based'
                all_questions.extend(role_mcqs)
                
                project_mcqs = generate_project_specific_mcqs(projects, qa_chain)
                for q in project_mcqs:
                    q['category'] = 'project_specific'
                all_questions.extend(project_mcqs)
                
                system_design_mcq = generate_system_design_mcq(projects, qa_chain)
                if system_design_mcq:
                    system_design_mcq['category'] = 'system_design'
                    all_questions.append(system_design_mcq)
                
                mcq_result = {
                    "success": True,
                    "role": role,
                    "projects": projects,
                    "questions": all_questions
                }
        
        # Combine all results
        response = {
            "success": True,
            "filename": file.filename,
            "achievements": achievements_result,
            "certifications": {
                "with_links": certs_with_links,
                "without_links": certs_without_links
            },
            "mcqs": mcq_result
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")
    
    finally:
        # Cleanup temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)