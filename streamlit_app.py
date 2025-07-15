import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import openai
from openai import OpenAI
import os

# Set page config
st.set_page_config(
    page_title="TalentFlow ATS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .similarity-high {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .similarity-medium {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .similarity-low {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .job-description-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'open_requisitions_df' not in st.session_state:
    st.session_state.open_requisitions_df = None
if 'filled_requisitions_df' not in st.session_state:
    st.session_state.filled_requisitions_df = None
if 'time_to_fill_df' not in st.session_state:
    st.session_state.time_to_fill_df = None
if 'filled_embeddings' not in st.session_state:
    st.session_state.filled_embeddings = None
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None

class OpenAIJobMatcher:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def get_embedding(self, text, model="text-embedding-3-small"):
        """Get embedding from OpenAI API"""
        try:
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error getting embedding: {str(e)}")
            return None
    
    def get_embeddings_batch(self, texts, model="text-embedding-3-small"):
        """Get embeddings for multiple texts"""
        embeddings = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(texts):
            embedding = self.get_embedding(text, model)
            if embedding:
                embeddings.append(embedding)
            else:
                # Use zero vector if embedding fails
                embeddings.append([0] * 1536)  # text-embedding-3-small dimension
            
            progress_bar.progress((i + 1) / len(texts))
        
        progress_bar.empty()
        return np.array(embeddings)

def load_csv_files():
    """Load CSV files from the repository"""
    try:
        # Load open requisitions
        open_req_df = pd.read_csv('open_requisitions.csv')
        st.session_state.open_requisitions_df = open_req_df
        
        # Load filled requisitions
        filled_req_df = pd.read_csv('filled_requisitions.csv')
        st.session_state.filled_requisitions_df = filled_req_df
        
        # Load time to fill
        ttf_df = pd.read_csv('time_to_fill.csv')
        st.session_state.time_to_fill_df = ttf_df
        
        return True, "CSV files loaded successfully!"
    except Exception as e:
        return False, f"Error loading CSV files: {str(e)}"

def create_filled_embeddings(matcher):
    """Create embeddings for all filled requisitions"""
    if st.session_state.filled_requisitions_df is None:
        return False, "Filled requisitions not loaded"
    
    try:
        job_descriptions = st.session_state.filled_requisitions_df['Job_Description'].tolist()
        
        with st.spinner("Creating embeddings for filled requisitions..."):
            embeddings = matcher.get_embeddings_batch(job_descriptions)
            st.session_state.filled_embeddings = embeddings
        
        return True, f"Created embeddings for {len(embeddings)} filled requisitions"
    except Exception as e:
        return False, f"Error creating embeddings: {str(e)}"

def find_similar_filled_jobs(matcher, selected_job_description, top_k=10):
    """Find similar filled jobs using cosine similarity"""
    if st.session_state.filled_embeddings is None:
        return None, "Filled embeddings not created"
    
    try:
        # Get embedding for selected job
        with st.spinner("Getting embedding for selected job..."):
            selected_embedding = matcher.get_embedding(selected_job_description)
        
        if selected_embedding is None:
            return None, "Failed to get embedding for selected job"
        
        # Calculate similarities
        selected_embedding = np.array(selected_embedding).reshape(1, -1)
        similarities = cosine_similarity(selected_embedding, st.session_state.filled_embeddings)[0]
        
        # Get top k similar jobs
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create results
        results = []
        for idx in top_indices:
            filled_req = st.session_state.filled_requisitions_df.iloc[idx]
            requisition_number = filled_req['Requisition_Number']
            
            # Get time to fill
            ttf_match = st.session_state.time_to_fill_df[
                st.session_state.time_to_fill_df['Requisition_Number'] == requisition_number
            ]
            
            time_to_fill = ttf_match['Time_to_Fill'].iloc[0] if not ttf_match.empty else "N/A"
            
            results.append({
                'Rank': len(results) + 1,
                'Requisition_Number': requisition_number,
                'Job_Title': filled_req['Job_Title'],
                'Job_Description': filled_req['Job_Description'],
                'Similarity_Score': float(similarities[idx]),
                'Time_to_Fill': time_to_fill
            })
        
        return pd.DataFrame(results), "Success"
    except Exception as e:
        return None, f"Error finding similar jobs: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 TalentFlow ATS</h1>
        <p>AI-Powered Job Matching with OpenAI Embeddings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # OpenAI API Key input
        st.subheader("🔑 OpenAI API Key")
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Your OpenAI API key for generating embeddings"
        )
        
        if api_key:
            try:
                st.session_state.openai_client = OpenAIJobMatcher(api_key)
                st.success("✅ OpenAI client initialized!")
            except Exception as e:
                st.error(f"❌ Error initializing OpenAI client: {str(e)}")
        
        # Load CSV files
        st.subheader("📁 Data Loading")
        if st.button("Load CSV Files"):
            success, message = load_csv_files()
            if success:
                st.success(message)
                
                # Show data summary
                st.write("**Data Summary:**")
                st.write(f"- Open Requisitions: {len(st.session_state.open_requisitions_df)}")
                st.write(f"- Filled Requisitions: {len(st.session_state.filled_requisitions_df)}")
                st.write(f"- Time to Fill Records: {len(st.session_state.time_to_fill_df)}")
            else:
                st.error(message)
        
        # Create embeddings for filled requisitions
        if (st.session_state.openai_client and 
            st.session_state.filled_requisitions_df is not None and 
            st.session_state.filled_embeddings is None):
            
            st.subheader("🧠 Embeddings")
            if st.button("Create Embeddings for Filled Requisitions"):
                success, message = create_filled_embeddings(st.session_state.openai_client)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Settings
        st.subheader("⚙️ Settings")
        top_k = st.slider("Number of similar jobs to find:", 5, 20, 10)
    
    # Main content
    if st.session_state.open_requisitions_df is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Select Open Requisition")
            
            # Create selectbox with requisition numbers
            requisition_options = st.session_state.open_requisitions_df['Requisition_Number'].tolist()
            selected_requisition = st.selectbox(
                "Choose a requisition number:",
                options=requisition_options,
                help="Select an open requisition to find similar filled positions"
            )
            
            # Display selected job description
            if selected_requisition:
                selected_row = st.session_state.open_requisitions_df[
                    st.session_state.open_requisitions_df['Requisition_Number'] == selected_requisition
                ]
                
                if not selected_row.empty:
                    job_description = selected_row['Job_Description'].iloc[0]
                    
                    st.subheader("📄 Job Description")
                    st.markdown(f"""
                    <div class="job-description-box">
                        <strong>Requisition:</strong> {selected_requisition}<br><br>
                        <strong>Description:</strong><br>
                        {job_description}
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔍 Analysis Controls")
            
            # Show status of embeddings
            if st.session_state.filled_embeddings is not None:
                st.success(f"✅ Embeddings ready for {len(st.session_state.filled_embeddings)} filled requisitions")
            else:
                st.warning("⚠️ Please create embeddings first (see sidebar)")
            
            # Analysis button
            if (st.session_state.openai_client and 
                st.session_state.filled_embeddings is not None and 
                selected_requisition):
                
                if st.button("🚀 Find Similar Filled Jobs", type="primary"):
                    selected_row = st.session_state.open_requisitions_df[
                        st.session_state.open_requisitions_df['Requisition_Number'] == selected_requisition
                    ]
                    job_description = selected_row['Job_Description'].iloc[0]
                    
                    results_df, status = find_similar_filled_jobs(
                        st.session_state.openai_client, 
                        job_description, 
                        top_k
                    )
                    
                    if results_df is not None:
                        # Display results
                        st.subheader("🎯 Similar Filled Jobs")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Matches", len(results_df))
                        with col2:
                            high_sim = len(results_df[results_df['Similarity_Score'] >= 0.8])
                            st.metric("High Similarity (≥80%)", high_sim)
                        with col3:
                            if 'Time_to_Fill' in results_df.columns:
                                numeric_ttf = pd.to_numeric(results_df['Time_to_Fill'], errors='coerce')
                                avg_ttf = numeric_ttf.mean()
                                if not pd.isna(avg_ttf):
                                    st.metric("Avg Time to Fill", f"{avg_ttf:.0f} days")
                                else:
                                    st.metric("Avg Time to Fill", "N/A")
                        with col4:
                            max_sim = results_df['Similarity_Score'].max()
                            st.metric("Best Match", f"{max_sim*100:.1f}%")
                        
                        # Results table
                        st.subheader("📊 Detailed Results")
                        
                        # Format the results for display
                        display_df = results_df.copy()
                        display_df['Similarity_Score'] = display_df['Similarity_Score'].apply(
                            lambda x: f"{x*100:.1f}%"
                        )
                        
                        # Show the table
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Job_Description": st.column_config.TextColumn(
                                    "Job Description",
                                    width="large"
                                )
                            }
                        )
                        
                        # Visualization
                        st.subheader("📈 Similarity Analysis")
                        
                        # Create similarity chart
                        fig = px.bar(
                            results_df,
                            x='Requisition_Number',
                            y='Similarity_Score',
                            color='Similarity_Score',
                            color_continuous_scale='RdYlGn',
                            title='Job Similarity Scores',
                            labels={'Similarity_Score': 'Similarity Score', 'Requisition_Number': 'Requisition Number'}
                        )
                        fig.update_layout(
                            showlegend=False,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Time to fill analysis if available
                        if 'Time_to_Fill' in results_df.columns:
                            numeric_results = results_df.copy()
                            numeric_results['Time_to_Fill_Numeric'] = pd.to_numeric(
                                numeric_results['Time_to_Fill'], errors='coerce'
                            )
                            
                            # Filter out N/A values
                            valid_data = numeric_results.dropna(subset=['Time_to_Fill_Numeric'])
                            
                            if len(valid_data) > 0:
                                st.subheader("⏱️ Time to Fill Analysis")
                                
                                # Scatter plot
                                fig2 = px.scatter(
                                    valid_data,
                                    x='Similarity_Score',
                                    y='Time_to_Fill_Numeric',
                                    size='Similarity_Score',
                                    hover_data=['Requisition_Number', 'Job_Title'],
                                    title='Similarity Score vs Time to Fill',
                                    labels={'Similarity_Score': 'Similarity Score', 'Time_to_Fill_Numeric': 'Time to Fill (Days)'}
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # Insights
                                st.subheader("💡 Insights")
                                high_sim_jobs = valid_data[valid_data['Similarity_Score'] >= 0.8]
                                if len(high_sim_jobs) > 0:
                                    avg_ttf_high_sim = high_sim_jobs['Time_to_Fill_Numeric'].mean()
                                    st.info(f"**High similarity jobs (≥80%)** took an average of **{avg_ttf_high_sim:.0f} days** to fill.")
                                
                                if len(valid_data) > 2:
                                    correlation = valid_data['Similarity_Score'].corr(valid_data['Time_to_Fill_Numeric'])
                                    if abs(correlation) > 0.3:
                                        direction = "positive" if correlation > 0 else "negative"
                                        st.info(f"There's a **{direction} correlation** ({correlation:.2f}) between similarity and time to fill.")
                    else:
                        st.error(f"Error: {status}")
    else:
        st.info("👈 Please load CSV files from the sidebar to get started.")
        
        # Show expected file structure
        st.subheader("📋 Expected File Structure")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**open_requisitions.csv:**")
            st.code("""Requisition_Number,Job_Description
REQ-2024-001,"Senior Software Engineer..."
REQ-2024-002,"Data Scientist..."
            """)
        
        with col2:
            st.write("**filled_requisitions.csv:**")
            st.code("""Requisition_Number,Job_Title,Job_Description
REQ-2023-001,"Software Engineer","Full stack dev..."
REQ-2023-002,"Data Analyst","Analytics role..."
            """)
        
        with col3:
            st.write("**time_to_fill.csv:**")
            st.code("""Requisition_Number,Time_to_Fill
REQ-2023-001,45
REQ-2023-002,32
            """)

if __name__ == "__main__":
    main()
