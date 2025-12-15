import json
import re
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ========================= PROJECT EXTRACTION =========================

def identify_project_section(text, qa_chain):
    """
    Identify and extract the projects section from resume.
    """
    prompt = f"""
You are analyzing a resume. Your task is to IDENTIFY and EXTRACT the complete section that contains projects.

Look for sections with headers like:
- Projects
- Personal Projects
- Academic Projects
- Work Projects
- Portfolio

Return ONLY valid JSON in this exact format (no extra text):
{{
  "project_section": {{
    "found": true,
    "content": "full extracted section text here"
  }}
}}

If no project section found:
{{
  "project_section": {{
    "found": false,
    "content": ""
  }}
}}

Resume text:
{text}
"""
    
    response = qa_chain.run(prompt)
    
    print(f"\n[DEBUG] Raw project section response:\n{response}\n")
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        
        # Auto-correct if content exists but found=false
        if result["project_section"]["content"] and not result["project_section"]["found"]:
            result["project_section"]["found"] = True
            print("[DEBUG] Auto-corrected project_section found flag")
        
        return result
    except Exception as e:
        print(f"[ERROR] JSON parsing error: {e}")
        return {"project_section": {"found": False, "content": ""}}


def extract_top_5_projects(section_text, qa_chain):
    """
    Extract top 5 projects (by order in resume) with their details.
    """
    prompt = f"""
From this project section, extract the FIRST 5 PROJECTS in order of appearance.

For each project, extract:
- title: Project name/title
- description: Brief description of what the project does
- technologies: List of technologies/libraries/frameworks used (e.g., Python, React, TensorFlow, FAISS, FastAPI)
- domain: Domain/category (e.g., Web Development, Machine Learning, Computer Vision, NLP, Full Stack)

Return ONLY valid JSON in this exact format (no extra text, no markdown):
{{
  "projects": [
    {{
      "title": "Project Name",
      "description": "What the project does",
      "technologies": ["Python", "FastAPI", "FAISS"],
      "domain": "Machine Learning"
    }}
  ]
}}

Project section text:
{section_text}
"""
    
    response = qa_chain.run(prompt)
    
    print(f"\n[DEBUG] Raw project extraction response:\n{response}\n")
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        
        # Limit to 5 projects
        projects = result.get("projects", [])[:5]
        
        print(f"\n[DEBUG] Extracted {len(projects)} projects:")
        for idx, proj in enumerate(projects, 1):
            print(f"   {idx}. {proj.get('title', 'N/A')} - {proj.get('domain', 'N/A')}")
            print(f"      Tech: {', '.join(proj.get('technologies', []))}")
        
        return {"projects": projects}
    except Exception as e:
        print(f"[ERROR] JSON parsing error: {e}")
        return {"projects": []}


# ========================= MCQ GENERATION =========================

def generate_role_based_mcqs(role: str, qa_chain) -> List[Dict]:
    """
    Generate 2 MCQs based on the role requirements.
    Role examples: "Python Developer", "Full Stack Developer", "ML Engineer"
    """
    prompt = f"""
You are generating technical MCQ questions for a {role} position.

Generate 2 medium-difficulty MCQ questions that test fundamental concepts for this role.

Requirements:
- Questions should test core concepts (OOP, frameworks, best practices)
- Medium difficulty (not too basic, not too advanced)
- 4 options each
- Only 1 correct answer

Return ONLY valid JSON in this exact format:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option B",
      "explanation": "Brief explanation of why this is correct"
    }}
  ]
}}

Generate 2 questions for {role} role.
"""
    
    response = qa_chain.run(prompt)
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        return result.get("questions", [])[:2]  # Ensure only 2 questions
    except Exception as e:
        print(f"[ERROR] Role-based MCQ generation error: {e}")
        return []


def generate_project_specific_mcqs(projects: List[Dict], qa_chain) -> List[Dict]:
    """
    Generate 3 MCQs specific to the technologies/libraries used in projects.
    """
    # Create a summary of all technologies used
    all_techs = []
    project_summaries = []
    
    for proj in projects:
        all_techs.extend(proj.get("technologies", []))
        project_summaries.append(f"{proj['title']}: {proj['description']} (Tech: {', '.join(proj.get('technologies', []))})")
    
    tech_summary = ", ".join(list(set(all_techs)))
    projects_text = "\n".join(project_summaries)
    
    prompt = f"""
You are generating technical MCQ questions based on specific projects.

Projects:
{projects_text}

Technologies used: {tech_summary}

Generate 3 medium-difficulty MCQ questions that test:
- Specific library/framework usage from their projects
- Implementation details
- Best practices for the technologies they used

Example: If they used FAISS, ask about index types or similarity search.
Example: If they used React, ask about hooks or state management.

Requirements:
- Questions must be directly related to technologies they used
- Medium difficulty
- 4 options each
- Only 1 correct answer

Return ONLY valid JSON in this exact format:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option B",
      "explanation": "Brief explanation",
      "related_project": "Project title this question relates to"
    }}
  ]
}}

Generate 3 project-specific questions.
"""
    
    response = qa_chain.run(prompt)
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        return result.get("questions", [])[:3]  # Ensure only 3 questions
    except Exception as e:
        print(f"[ERROR] Project-specific MCQ generation error: {e}")
        return []


def generate_system_design_mcq(projects: List[Dict], qa_chain) -> Dict:
    """
    Generate 1 system design MCQ based on one of their projects.
    """
    # Pick the most complex/interesting project (first one for now)
    if not projects:
        return {}
    
    selected_project = projects[0]
    
    prompt = f"""
You are generating a system design MCQ question based on a project.

Project: {selected_project['title']}
Description: {selected_project['description']}
Technologies: {', '.join(selected_project.get('technologies', []))}
Domain: {selected_project.get('domain', '')}

Generate 1 medium-difficulty system design MCQ that tests:
- Architectural decisions
- Scalability considerations
- Design patterns
- Technology choices

The question should be related to how they could improve/scale this project.

Requirements:
- System design focused (not implementation details)
- Medium difficulty
- 4 options each
- Only 1 correct answer

Return ONLY valid JSON in this exact format:
{{
  "question": "Question text here?",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": "Option B",
  "explanation": "Brief explanation",
  "related_project": "{selected_project['title']}"
}}

Generate 1 system design question.
"""
    
    response = qa_chain.run(prompt)
    
    try:
        response_clean = re.sub(r"```json|```", "", response).strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_clean = response_clean[start_idx:end_idx+1]
        
        result = json.loads(response_clean)
        return result
    except Exception as e:
        print(f"[ERROR] System design MCQ generation error: {e}")
        return {}


# ========================= MAIN ORCHESTRATOR =========================

def generate_all_mcqs(text, qa_chain, role: str = "Python Developer") -> Dict[str, Any]:
    """
    Main function to generate all 6 MCQs for Feature 3.
    
    Args:
        text: Full resume text
        qa_chain: LangChain QA chain
        role: Target role (default: "Python Developer")
    
    Returns:
        Dictionary with all questions and metadata
    """
    print("\n" + "="*80)
    print("FEATURE 3: PROJECT-BASED MCQ GENERATION")
    print("="*80)
    print(f"\nTarget Role: {role}")
    
    # Step 1: Identify project section
    print("\n🔍 Step 1: Identifying project section...")
    project_section_data = identify_project_section(text, qa_chain)
    
    if not project_section_data["project_section"]["found"]:
        print("❌ No project section found in resume")
        return {
            "success": False,
            "error": "No project section found",
            "questions": []
        }
    
    print("✅ Project section found!")
    project_section = project_section_data["project_section"]["content"]
    
    # Step 2: Extract top 5 projects
    print("\n🔍 Step 2: Extracting top 5 projects...")
    projects_data = extract_top_5_projects(project_section, qa_chain)
    projects = projects_data.get("projects", [])
    
    if not projects:
        print("❌ No projects extracted")
        return {
            "success": False,
            "error": "No projects extracted from section",
            "questions": []
        }
    
    print(f"✅ Extracted {len(projects)} projects")
    
    # Step 3: Generate MCQs
    print("\n🔍 Step 3: Generating MCQs...")
    
    all_questions = []
    
    # 2 Role-based MCQs
    print("\n   Generating 2 role-based MCQs...")
    role_mcqs = generate_role_based_mcqs(role, qa_chain)
    for q in role_mcqs:
        q['category'] = 'role_based'
    all_questions.extend(role_mcqs)
    print(f"   ✅ Generated {len(role_mcqs)} role-based MCQs")
    
    # 3 Project-specific MCQs
    print("\n   Generating 3 project-specific MCQs...")
    project_mcqs = generate_project_specific_mcqs(projects, qa_chain)
    for q in project_mcqs:
        q['category'] = 'project_specific'
    all_questions.extend(project_mcqs)
    print(f"   ✅ Generated {len(project_mcqs)} project-specific MCQs")
    
    # 1 System design MCQ
    print("\n   Generating 1 system design MCQ...")
    system_design_mcq = generate_system_design_mcq(projects, qa_chain)
    if system_design_mcq:
        system_design_mcq['category'] = 'system_design'
        all_questions.append(system_design_mcq)
        print(f"   ✅ Generated 1 system design MCQ")
    
    print(f"\n✅ Total MCQs generated: {len(all_questions)}/6")
    
    return {
        "success": True,
        "role": role,
        "projects": projects,
        "questions": all_questions,
        "total_questions": len(all_questions)
    }


def save_mcqs_to_file(mcq_data: Dict, filename: str = "generated_mcqs.json"):
    """
    Save generated MCQs to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(mcq_data, f, indent=2)
    print(f"\n✅ MCQs saved to {filename}")


# ========================= EXAMPLE USAGE =========================

if __name__ == "__main__":
    # This is an example - in actual use, you'll call this from your main app
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import PyPDF2
    
    def extract_text_from_pdf(file_path):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    
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
    
    # Example usage
    resume_path = "resume.pdf"
    text = extract_text_from_pdf(resume_path)
    vectorstore = create_vector_store(text)
    qa_chain = create_qa_chain(vectorstore)
    
    # Generate MCQs for a Python Developer role
    mcq_data = generate_all_mcqs(text, qa_chain, role="Python Developer")
    
    if mcq_data["success"]:
        save_mcqs_to_file(mcq_data)
        
        # Display questions
        print("\n" + "="*80)
        print("GENERATED MCQs PREVIEW")
        print("="*80)
        for idx, q in enumerate(mcq_data["questions"], 1):
            print(f"\nQ{idx} [{q['category']}]: {q['question']}")
            for opt in q['options']:
                print(f"   {opt}")
            print(f"   ✓ Correct: {q['correct_answer']}")
    else:
        print(f"\n❌ Error: {mcq_data['error']}")