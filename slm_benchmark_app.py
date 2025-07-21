"""
WebApp made solely so SugarLab community members can easily compare and analyze responses from Fine-Tuned versions of LlaMA-135M. This will be helpful in selecting which model to use for the speak activity.
Script made with AI, there may be mistakes. Use at your own caution.
"""

import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import re

st.set_page_config(
    page_title="SLM Benchmark Comparison Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 2rem;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .response-box {
        border-left: 4px solid #3498db;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    .student-question {
        font-weight: bold;
        color: #e74c3c;
        margin-bottom: 10px;
    }
    .teacher-response {
        color: #27ae60;
        margin-left: 20px;
    }
    .metric-card {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px;
    }
    .answer-source {
        background-color: #e8f4f8;
        border: 1px solid #3498db;
        border-radius: 5px;
        padding: 5px 10px;
        margin: 5px 0;
        font-size: 0.9em;
        color: #2c3e50;
        font-family: 'Courier New', monospace;
    }
    .sticky-question {
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        z-index: 1000;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        margin: 0 -20px 30px -20px;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 3px solid #5a67d8;
    }
    .question-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .question-text {
        font-size: 1.4rem;
        font-weight: 500;
        line-height: 1.6;
        text-align: center;
        background: rgba(255,255,255,0.15);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffd700;
        backdrop-filter: blur(10px);
    }
    .answers-container {
        margin-top: 20px;
    }
    .stSelectbox > div[data-baseweb="select"] {
        background-color: #f0f8ff;
        border: 2px solid #3498db;
        border-radius: 10px;
        font-size: 1.1em;
        font-weight: 600;
    }
    .stSelectbox > div[data-baseweb="select"]:hover {
        border-color: #2980b9;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
    .question-selector-container {
        background: linear-gradient(135deg, #e8f4f8 0%, #d5e9f0 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #3498db;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .question-selector-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 15px;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class SLMBenchmarkApp:
    def __init__(self):
        # Use relative path that works both locally and when deployed
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.models_data = {}
        self.load_all_models()
        
    def extract_model_info(self, model_name, model_type, folder_name):
        if "MebinThattil/" in model_name:
            clean_name = model_name.split("MebinThattil/")[-1]
        elif ".gguf" in model_name:
            clean_name = os.path.basename(model_name).replace('.gguf', '')
        else:
            clean_name = model_name
            
        if "Claude" in folder_name:
            if "RUN1" in folder_name:
                category = "Claude Distill RUN1"
            else:
                category = "Claude Distill RUN2"
        elif "Gemini" in folder_name:
            category = "Gemini 2.5 PRO Distill"
        elif "Educational_Dataset_ConversationAware" in folder_name:
            category = "Educational ConversationAware"
        elif "Educational_Dataset_GH_Conversation_Unaware" in folder_name:
            category = "Educational ConversationUnaware"
        else:
            category = "Other"
            
        return {
            'display_name': clean_name,
            'category': category,
            'quantization': model_type
        }

    def load_json_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {filepath}: {str(e)}")
            return None

    def load_all_models(self):
        folders = [
            "Distill_Claude_RUN1",
            "Distill_Claude_RUN2", 
            "Distill_Gemini_2.5PRO_ConversationAware",
            "Educational_Dataset_ConversationAware",
            "Educational_Dataset_GH_Conversation_Unaware"
        ]
        
        for folder in folders:
            folder_path = os.path.join(self.base_path, folder)
            if os.path.exists(folder_path):
                self.load_folder_models(folder_path, folder)

    def find_answer_folders(self, folder_path):
        answer_folders = []
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                if any(keyword in dir_name.lower() for keyword in ['answer', 'benchmark']):
                    answer_folders.append(os.path.join(root, dir_name))
        return answer_folders

    def load_folder_models(self, folder_path, folder_name):
        answer_folders = self.find_answer_folders(folder_path)
        
        for answer_folder in answer_folders:
            quant_folders = ['Answers-Unquantized', 'Answer - Unquantized', 'Answers - Unquantized',
                           'Answers-GGUF', 'Answer-GGUF', 'Answers - GGUF', 
                           'Answers-GGUF-Q4', 'Answers - GGUF-Q4']
            
            for quant_folder in quant_folders:
                quant_path = os.path.join(answer_folder, quant_folder)
                if os.path.exists(quant_path):
                    self.load_quantization_models(quant_path, folder_name, quant_folder)

    def load_quantization_models(self, quant_path, folder_name, quant_type):
        json_files = [f for f in os.listdir(quant_path) if f.endswith('.json')]
        
        for json_file in json_files:
            file_path = os.path.join(quant_path, json_file)
            data = self.load_json_file(file_path)
            
            if data and 'metadata' in data and 'results' in data:
                model_info = self.extract_model_info(
                    data['metadata']['model_name'],
                    quant_type,
                    folder_name
                )
                
                model_key = f"{folder_name}_{quant_type}_{json_file}"
                
                self.models_data[model_key] = {
                    'metadata': data['metadata'],
                    'results': data['results'],
                    'info': model_info,
                    'file_name': json_file,
                    'folder': folder_name,
                    'quantization': quant_type
                }

    def get_model_categories(self):
        categories = set()
        for model_data in self.models_data.values():
            categories.add(model_data['info']['category'])
        return sorted(list(categories))

    def get_quantization_types(self):
        quant_types = set()
        for model_data in self.models_data.values():
            quant_types.add(model_data['quantization'])
        return sorted(list(quant_types))

    def create_model_overview(self):
        st.markdown('<h1 class="main-header">🤖 SLM Benchmark Comparison Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{len(self.models_data)}</h2>
                <p>Total Models</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{len(self.get_model_categories())}</h2>
                <p>Model Categories</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{len(self.get_quantization_types())}</h2>
                <p>Quantization Types</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            total_questions = len(list(self.models_data.values())[0]['results']) if self.models_data else 0
            st.markdown(f"""
            <div class="metric-card">
                <h2>{total_questions}</h2>
                <p>Questions per Model</p>
            </div>
            """, unsafe_allow_html=True)

    def create_model_comparison_table(self):
        st.subheader("📊 Model Comparison Table")
        
        model_comparison = []
        for key, model_data in self.models_data.items():
            metadata = model_data['metadata']
            info = model_data['info']
            
            file_name = model_data['file_name'].replace('.json', '')
            
            model_comparison.append({
                'Model Display Name': info['display_name'],
                'Category': info['category'],
                'Quantization': model_data['quantization'],
                'Answer File': file_name,
                'Max Tokens': metadata['generation_parameters'].get('max_tokens', 'N/A'),
                'Temperature': metadata['generation_parameters'].get('temperature', 'N/A'),
                'Top P': metadata['generation_parameters'].get('top_p', 'N/A'),
                'Device': metadata['processing_info'].get('device', 'N/A'),
                'Total Questions': metadata['processing_info'].get('total_questions', 'N/A')
            })
        
        df = pd.DataFrame(model_comparison)
        st.dataframe(df, use_container_width=True)
        
        return df

    def create_response_comparison(self):
        st.subheader("🔍 Response Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Filter Models")
            selected_categories = st.multiselect(
                "Select Categories",
                self.get_model_categories(),
                default=self.get_model_categories()
            )
            
            all_answer_files = set()
            for model_data in self.models_data.values():
                answer_file = model_data['file_name'].replace('.json', '')
                all_answer_files.add(answer_file)
            
            selected_answer_files = st.multiselect(
                "Select Answer Files (Optional)",
                sorted(list(all_answer_files)),
                help="Filter by specific answer files like Answer1, Answer1_1, Answer2, etc."
            )
        
        with col2:
            st.markdown("""
                        <br><br><br>
            <div class="question-selector-container">
                <div class="question-selector-title"> Question Selection</div>
            </div>
            """, unsafe_allow_html=True)
            
            if self.models_data:
                total_questions = len(list(self.models_data.values())[0]['results'])
                question_num = st.selectbox(
                    "Select Question Number",
                    range(1, total_questions + 1),
                    help=f"Choose from {total_questions} available questions",
                    key="question_selector"
                )
        
        filtered_models = {}
        for key, model_data in self.models_data.items():
            answer_file = model_data['file_name'].replace('.json', '')
            
            if (model_data['info']['category'] in selected_categories and
                (not selected_answer_files or answer_file in selected_answer_files)):
                filtered_models[key] = model_data
        
        if filtered_models and question_num:
            sample_model = list(filtered_models.values())[0]
            question_text = sample_model['results'][question_num-1]['student']
            
            st.markdown(f"""
            <div class="sticky-question">
                <div class="question-header">Question {question_num}</div>
                <div class="question-text">{question_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="answers-container">', unsafe_allow_html=True)
            
            model_hierarchy = {}
            for key, model_data in filtered_models.items():
                category = model_data['info']['category']
                quantization = model_data['quantization']
                answer_file = model_data['file_name'].replace('.json', '')
                
                if category not in model_hierarchy:
                    model_hierarchy[category] = {}
                if quantization not in model_hierarchy[category]:
                    model_hierarchy[category][quantization] = {}
                if answer_file not in model_hierarchy[category][quantization]:
                    model_hierarchy[category][quantization][answer_file] = []
                
                model_hierarchy[category][quantization][answer_file].append((key, model_data))
            
            for category, quantization_groups in sorted(model_hierarchy.items()):
                st.markdown(f"## ➡️ {category}")
                
                quantization_order = [
                    'Answers-Unquantized', 'Answer - Unquantized', 'Answers - Unquantized',
                    'Answers-GGUF', 'Answer-GGUF', 'Answers - GGUF',
                    'Answers-GGUF-Q4', 'Answers - GGUF-Q4'
                ]
                
                def get_quantization_priority(quant_type):
                    for i, pattern in enumerate(quantization_order):
                        if pattern == quant_type:
                            return i
                    return len(quantization_order)
                
                sorted_quantization_groups = sorted(
                    quantization_groups.items(),
                    key=lambda x: get_quantization_priority(x[0])
                )
                
                for quantization, answer_groups in sorted_quantization_groups:
                    st.markdown(f"### 🔵 {quantization}")
                    
                    answer_items = list(sorted(answer_groups.items()))
                    models_per_row = 3
                    
                    for i in range(0, len(answer_items), models_per_row):
                        cols = st.columns(min(models_per_row, len(answer_items) - i))
                        for j, col in enumerate(cols):
                            if i + j < len(answer_items):
                                answer_file, models = answer_items[i + j]
                                with col:
                                    st.markdown(f"#### 📄 {answer_file}")
                                    for key, model_data in models:
                                        self.display_model_response(model_data, question_num-1)
                    
                    st.divider()
                
                st.markdown("---")
            
            st.markdown('</div>', unsafe_allow_html=True)

    def display_model_response(self, model_data, question_idx):
        info = model_data['info']
        result = model_data['results'][question_idx]
        metadata = model_data['metadata']
        
        file_name = model_data['file_name'].replace('.json', '')
        st.markdown(f'<div class="answer-source">📄 {file_name}</div>', 
                   unsafe_allow_html=True)
        
        teacher_response = result.get('teacher', 'No response')
        st.markdown(f'<div class="response-box">'
                   f'<div class="teacher-response">{teacher_response}</div>'
                   f'</div>', unsafe_allow_html=True)
        
        with st.expander("Details"):
            st.json({
                'Model': info['display_name'],
                'Category': info['category'],
                'Quantization': model_data['quantization'],
                'Source File': model_data['file_name'],
                'Generation Parameters': metadata.get('generation_parameters', {}),
                'Processing Info': metadata.get('processing_info', {})
            })

    def create_analytics_dashboard(self):
        st.subheader("📈 Analytics Dashboard")
        
        response_lengths = defaultdict(list)
        model_names = []
        categories = []
        quantizations = []
        
        for key, model_data in self.models_data.items():
            model_names.append(model_data['info']['display_name'])
            categories.append(model_data['info']['category'])
            quantizations.append(model_data['quantization'])
            
            lengths = [len(result.get('teacher', '')) for result in model_data['results']]
            response_lengths[key] = lengths
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_lengths = {key: sum(lengths)/len(lengths) for key, lengths in response_lengths.items()}
            
            fig_lengths = go.Figure(data=[
                go.Bar(x=list(avg_lengths.keys()), y=list(avg_lengths.values()))
            ])
            fig_lengths.update_layout(
                title="Average Response Length by Model",
                xaxis_title="Model",
                yaxis_title="Average Character Count",
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig_lengths, use_container_width=True)
        
        with col2:
            all_lengths = []
            all_models = []
            
            for key, lengths in response_lengths.items():
                model_name = self.models_data[key]['info']['display_name']
                all_lengths.extend(lengths)
                all_models.extend([model_name] * len(lengths))
            
            df_lengths = pd.DataFrame({
                'Model': all_models,
                'Response Length': all_lengths
            })
            
            fig_dist = px.box(df_lengths, x='Model', y='Response Length',
                            title="Response Length Distribution")
            fig_dist.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig_dist, use_container_width=True)

    def create_model_deep_dive(self):
        st.subheader("🔬 Model Deep Dive")
        
        model_keys = list(self.models_data.keys())
        model_labels = [f"{self.models_data[key]['info']['display_name']} ({self.models_data[key]['quantization']}) - {self.models_data[key]['file_name'].replace('.json', '')}" 
                       for key in model_keys]
        
        selected_idx = st.selectbox("Select Model for Deep Dive", 
                                   range(len(model_labels)),
                                   format_func=lambda x: model_labels[x])
        
        if selected_idx is not None:
            selected_key = model_keys[selected_idx]
            model_data = self.models_data[selected_key]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Model Information")
                st.json({
                    'Display Name': model_data['info']['display_name'],
                    'Category': model_data['info']['category'],
                    'Quantization': model_data['quantization'],
                    'Answer File': model_data['file_name'].replace('.json', ''),
                    'Full File Path': model_data['file_name']
                })
                
                st.markdown("### Generation Parameters")
                st.json(model_data['metadata'].get('generation_parameters', {}))
            
            with col2:
                st.markdown("### All Responses")
                
                for i, result in enumerate(model_data['results'], 1):
                    with st.expander(f"Question {i}: {result['student'][:50]}..."):
                        st.markdown(f"**Student:** {result['student']}")
                        st.markdown(f"**Teacher:** {result.get('teacher', 'No response')}")

    def run(self):
        st.sidebar.title("📋 Navigation")
        page = st.sidebar.selectbox("Select Page", [
            "🔍 Response Comparison",
            "🔬 Deep Dive",
            "📈 Analytics"
        ])
        
        if len(self.models_data) == 0:
            st.error("No model data found! Please check your file paths.")
            return
        
        if page == "🔍 Response Comparison":
            self.create_response_comparison()
        elif page == "🔬 Deep Dive":
            self.create_model_deep_dive()
        elif page == "📈 Analytics":
            self.create_analytics_dashboard()

if __name__ == "__main__":
    app = SLMBenchmarkApp()
    app.run()
