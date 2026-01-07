
import streamlit as st
import google.generativeai as genai
import json
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import io
import plotly.graph_objects as go
import pandas as pd
import re

# Page Config
st.set_page_config(
    page_title="Sovereign AI T-Class Evaluator 2.0 (Auto-Spec)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .main_header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub_header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 1.5rem;
        font-weight: 500;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #f9fafb;
        margin-top: 20px;
        margin-bottom: 20px;
        box_shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #6B7280;
        font-size: 0.9rem;
    }
    .footer a {
        color: #2563EB;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# Helper Functions
def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def fetch_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

@st.cache_data(show_spinner=True)
def run_gemini_analysis(content_text, content_source):
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except FileNotFoundError:
        st.error("Secrets not found. Please set GEMINI_API_KEY in .streamlit/secrets.toml or your deployment settings.")
        return None, None
    except KeyError:
        st.error("GEMINI_API_KEY not set in secrets.")
        return None, None

    MODEL_NAME = "gemini-3-pro-preview"
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    
    system_prompt = """
# Role
당신은 대한민국 'Sovereign AI T-Class 2.0' 표준에 의거하여, AI 모델의 기술 주권 등급(T0 ~ T6)을 엄격하게 판정하고, 기술적 도약을 위한 조언을 제공하는 [수석 AI 주권 컨설턴트]입니다.

# Objective
1. 제공된 문서(Raw Text)를 정밀 분석하여, AI 모델의 핵심 명세(Spec)를 추출하십시오.
2. 추출된 정보를 바탕으로 엄격하게 등급을 판정하고, 전문적인 **마크다운(Markdown) 리포트**를 작성하십시오.
3. 마지막으로, 방사형 차트 생성을 위한 **주권 점수(Sovereignty Score)**를 JSON 형식으로 숨김(Comment) 처리 없이 정확하게 출력하십시오.

# Critical Assessment Rules (판정 대원칙)
1. **가중치(Weights) 원천 불가침의 원칙**: 
   - **타 모델의 가중치를 1%라도 상속받거나 추가 학습(CPT, SFT)했다면 무조건 T2 이하로 판정합니다.** 성능이 아무리 좋아도 예외는 없습니다.
   - T4 이상의 필수 조건은 **"Random Initialization(무작위 초기화) 상태에서 100% 자체 데이터로 학습(From Scratch)"** 한 경우뿐입니다.

2. **T4 vs T5 구분 (엔지니어링 실체)**:
   - **T4 (From Scratch)**: 아키텍처는 참조했을 수 있으나, 가중치는 무조건 0(Random Initialization)부터 자체 학습한 경우에만 해당합니다.
   - **T5 (Native)**: 독자적인 연산 그래프(Topology)를 설계하고 한국어 전용 토크나이저를 처음부터 구축한 경우에만 해당합니다. (재건축)

# T-Class 2.0 Grading Criteria (등급 기준표)

## [그룹 A] 의존 및 과도기 (타사 가중치 사용)
- **T0 (API Wrapper)**: 모델 없음. 빅테크 API(GPT, Claude 등) 호출.
- **T1 (Fine-Tuner)**: 가중치 비공개(Closed) 모델을 가져와 미세조정.
- **T2 (CPT/SFT)**: 가중치 공개(Open) 모델(Llama, Mistral 등)을 가져와 추가 학습. (한국어 패치 등)
- **T3 (Expanded/Merge)**: 오픈 웨이트 모델끼리 병합하거나 레이어를 복사(DUS)하여 개조. (리모델링)

## [그룹 B] 소버린 AI (가중치 100% 자체 학습)
- **T4 (From Scratch)**: 오픈소스 아키텍처(설계도)를 참고했으나, 가중치는 0부터 직접 학습.
    - *T4-1 (Adopter)*: 설정값(Config)까지 원본과 동일.
    - *T4-2 (Scaler)*: 레이어 확장 등 설정값 변경 및 최적화 수행.
- **T5 (Native Arch)**: 독자적인 블록 구조 설계(Code 변경) + 한국어 Native 토크나이저 구축. (호환되지 않는 독자 모델)
- **T6 (Full-Stack)**: T5 등급 모델 + 국산 NPU 구동 + 국산 클라우드 인프라.

# Output Format (마크다운 리포트 + JSON 데이터)

## Part 1: Markdwon Report
반드시 다음 구조로 작성하십시오.

## 🏆 Sovereign AI T-Class Evaluation Report

### 1. 등급 판정 (Decision)
# [T등급] (예: T4-2. Scaler)
> **"판정 핵심 요약 한 줄 (예: Llama 3 아키텍처를 차용했으나, 가중치를 3T 토큰으로 처음부터 학습하여 T4-1로 판정됨)"**

### 2. 상세 스펙 분석 (Technical Analysis)
| 평가 항목 | 추출 내용 | 분석 및 판정 |
| :--- | :--- | :--- |
| **기반 모델 (Base Model)** | (예: None - Random Init) | (예: 가중치 의존성 없음 (Pass)) |
| **학습 방식 (Training)** | (예: Pre-training from scratch) | (예: Sovereign AI 기준 충족) |
| **아키텍처 (Architecture)** | (예: LlamaForCausalLM) | (예: 표준 아키텍처 사용 (T4)) |
| **토크나이저 (Tokenizer)** | (예: Llama-3 Tokenizer) | (예: 타사 토크나이저 재사용) |
| **인프라 (Infrastructure)** | (예: AWS H100 Cluster) | (예: 외산 인프라 사용) |

### 3. 심층 평가 (Deep Dive)
- **가중치 주권 (Weight Sovereignty)**: (가중치 학습 과정에 대한 상세 분석)
- **기술 자립도 (Tech Independence)**: (아키텍처 및 원천 기술 확보 수준 평가)

---
__JSON_START__
{
  "weight_score": 0~10점 (가중치 원천성, T4 이상은 10점),
  "arch_score": 0~10점 (아키텍처 독자성, T5는 10점),
  "tokenizer_score": 0~10점 (언어 처리 독자성),
  "data_score": 0~10점 (학습 데이터 자립도),
  "infra_score": 0~10점 (인프라 자립도)
}
__JSON_END__
"""
    
    user_input_data = f"""
# Evaluation Target Context
- Source: {content_source}
- Content:
{content_text[:50000]} 
(Content Truncated if > 50k chars for efficiency)
"""
    
    response = model.generate_content(
        contents=[system_prompt, user_input_data]
    )
    return response.text, MODEL_NAME

def make_radar_chart(scores):
    categories = ['Weight Origin', 'Architecture', 'Tokenizer', 'Training Data', 'Infrastructure']
    values = [
        scores.get('weight_score', 0),
        scores.get('arch_score', 0),
        scores.get('tokenizer_score', 0),
        scores.get('data_score', 0),
        scores.get('infra_score', 0)
    ]
    
    # Close the loop
    categories = [*categories, categories[0]]
    values = [*values, values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Sovereignty Score',
                line_color='#1E3A8A',
                fillcolor='rgba(30, 58, 138, 0.2)'
            )
        ],
        layout=go.Layout(
            title=go.layout.Title(text='🛡️ AI Sovereignty Radar Chart'),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=False
        )
    )
    return fig

# Application Title
st.markdown('<div class="main_header">Sovereign AI T-Class Evaluator 2.0</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666;">Auto-Spec Analysis from Technical Reports & Spec Sheets</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔑 Configuration")
    
    # Active Model Display
    st.info(f"⚡ Active Engine:\ngemini-3-pro-preview")
    
    st.markdown("---")

    st.header("📋 T-Class 2.0 Criteria")
    st.info("""
    **Group A: Dependent**
    - T0: API Wrapper
    - T1: Fine-Tuner
    - T2: CPT/SFT
    - T3: Expanded/Merge
    
    **Group B: Sovereign**
    - T4: From Scratch
    - T5: Native Arch
    - T6: Full-Stack
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("**Created by epoko77**")
    st.markdown("[GitHub Repository](https://github.com/epoko77-ai/sovereign-ai-evaluator)")

# Main Content
col1, col2 = st.columns([1, 1])

# Input Column
with col1:
    st.subheader("📂 Import Document")
    
    tab1, tab2 = st.tabs(["📄 PDF Upload", "🔗 Web Link"])
    
    extracted_text = None
    source_name = None
    
    with tab1:
        uploaded_file = st.file_uploader("Upload Technical Report (PDF)", type="pdf")
        if uploaded_file is not None:
            if st.button("Read PDF", key="read_pdf"):
                with st.spinner("Extracting text from PDF..."):
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    source_name = f"PDF: {uploaded_file.name}"
                    if extracted_text:
                        st.success("Text extracted successfully!")
                        st.session_state['extracted_text'] = extracted_text
                        st.session_state['source_name'] = source_name
    
    with tab2:
        url_input = st.text_input("Enter Document URL (GitHub/HuggingFace/Blog)")
        if st.button("Fetch URL", key="fetch_url"):
            if url_input:
                with st.spinner("Fetching content from URL..."):
                    extracted_text = fetch_text_from_url(url_input)
                    source_name = f"URL: {url_input}"
                    if extracted_text:
                        st.success("Content fetched successfully!")
                        st.session_state['extracted_text'] = extracted_text
                        st.session_state['source_name'] = source_name
            else:
                st.warning("Please enter a URL.")

    # Show Preview of Text if available in session state
    if 'extracted_text' in st.session_state:
        st.markdown("##### 📝 Content Preview")
        st.text_area("Raw Text", st.session_state['extracted_text'][:1000] + "...", height=150, disabled=True)
        
        if st.button("🚀 Run Auto-Analysis", type="primary"):
            with col2:
                try:
                    full_response, model_name = run_gemini_analysis(st.session_state['extracted_text'], st.session_state['source_name'])
                    
                    # Parse Split (Markdown vs JSON)
                    parts = full_response.split("__JSON_START__")
                    markdown_report = parts[0].strip()
                    
                    scores = {}
                    if len(parts) > 1:
                        json_part = parts[1].split("__JSON_END__")[0].strip()
                        try:
                            scores = json.loads(json_part)
                        except:
                            st.warning("Failed to parse Sovereignty Scores.")
                    
                    st.subheader("🔍 Analysis Output")
                    
                    # Display Result Card (Markdown)
                    with st.container(border=True):
                        st.markdown(markdown_report)
                    
                    # Display Radar Chart
                    if scores:
                        st.markdown("### 📊 Sovereignty Radar")
                        fig = make_radar_chart(scores)
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    <hr>
    <p>Created by <b>epoko77</b> | <a href="https://github.com/epoko77-ai/sovereign-ai-evaluator" target="_blank">GitHub Repository</a></p>
    <p>© 2026 Sovereign AI Research Lab. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
