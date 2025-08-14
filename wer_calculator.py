import streamlit as st
import jiwer
import Levenshtein
import pandas as pd
from datetime import datetime
import io
import os

# Apply monospace font to everything
st.markdown("""
<style>
    .stApp {
        font-family: 'Courier New', monospace;
    }
    .stTextInput > div > div > input {
        font-family: 'Courier New', monospace;
    }
    .stButton > button {
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Set page config
st.set_page_config(page_title="WER Calculator", layout="centered")

# Title
st.title("WER/MER Calculator")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single Calculation", "Batch Processing", "About"])

with tab1:
    # Input fields
    ground_truth = st.text_input("Ground Truth:", placeholder="Enter reference text...")
    hypothesis = st.text_input("Hypothesis:", placeholder="Enter hypothesis text...")

    # Calculate button
    if st.button("Calculate"):
        # Check for empty fields
        if not ground_truth.strip():
            st.error("Please enter ground truth text")
        elif not hypothesis.strip():
            st.error("Please enter hypothesis text")
        else:
            try:
                # Process with jiwer
                out = jiwer.process_words([ground_truth], [hypothesis])

                # Calculate Levenshtein distance
                lev_distance = Levenshtein.distance(ground_truth, hypothesis)
                lev_ratio = Levenshtein.ratio(ground_truth, hypothesis)

                # Display metrics
                st.subheader("Alignment Visualization:")
                
                # Use jiwer's visualization - display in code block to preserve formatting
                alignment_viz = jiwer.visualize_alignment(out)
                st.code(alignment_viz)
                
                st.subheader("Additional Metrics:")
                additional_metrics = f"""Levenshtein Distance: {lev_distance}
Levenshtein Ratio: {lev_ratio:.3f}"""
                st.code(additional_metrics)
                
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")

with tab2:
    st.subheader("Batch Processing")

    st.write("**All processing happens in your browser - no data is stored**")
    
    st.write("**Input Requirements:**")
    st.write("Upload a CSV file with the following required columns:")
    st.code("""- ground_truth: Reference text (what should have been said)
- hypothesis: Hypothesis text (what was actually transcribed)""")
    
    st.write("**Output:**")
    st.write("The output CSV will contain all original columns plus these new columns:")
    st.code("""- WER: Word Error Rate (0.0 = perfect, higher = more errors)
- MER: Match Error Rate 
- MER_accuracy: Accuracy as percentage (100 - MER*100)
- Lev_raw: Levenshtein distance (character-level edits)
- Lev_ratio: Levenshtein similarity ratio (0.0-1.0)""")
    
    st.write("**Output Filename Format:**")
    st.code("original_filename_results_YYYYMMDD_HHMMSS.csv")
    st.write("Example: `my_data.csv` → `my_data_results_20241201_143022.csv`")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            if 'ground_truth' not in df.columns or 'hypothesis' not in df.columns:
                st.error("CSV must contain 'ground_truth' and 'hypothesis' columns")
            else:
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Process Batch"):
                    # Initialize result lists
                    wer_scores = []
                    mer_scores = []
                    mer_accuracy_scores = []
                    lev_raw_scores = []
                    lev_ratio_scores = []
                    
                    # Process each row
                    progress_bar = st.progress(0)
                    for idx, row in df.iterrows():
                        # Handle NaN/empty values
                        gt = str(row['ground_truth']) if pd.notna(row['ground_truth']) else ""
                        hyp = str(row['hypothesis']) if pd.notna(row['hypothesis']) else ""
                        
                        # Skip completely empty rows
                        if not gt.strip() or not hyp.strip():
                            wer_scores.append(None)
                            mer_scores.append(None)
                            mer_accuracy_scores.append(None)
                            lev_raw_scores.append(None)
                            lev_ratio_scores.append(None)
                        else:
                            try:
                                # Process with jiwer
                                out = jiwer.process_words([gt], [hyp])
                                
                                # Calculate metrics
                                wer_scores.append(out.wer)
                                mer_scores.append(out.mer)
                                mer_accuracy_scores.append((1 - out.mer) * 100)  # Convert to percentage
                                lev_raw_scores.append(Levenshtein.distance(gt, hyp))
                                lev_ratio_scores.append(Levenshtein.ratio(gt, hyp))
                            except Exception as e:
                                st.warning(f"Error processing row {idx + 1}: {str(e)}")
                                wer_scores.append(None)
                                mer_scores.append(None)
                                mer_accuracy_scores.append(None)
                                lev_raw_scores.append(None)
                                lev_ratio_scores.append(None)
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Create results dataframe (keeps ALL original columns)
                    results_df = df.copy()
                    results_df['WER'] = wer_scores
                    results_df['MER'] = mer_scores
                    results_df['MER_accuracy'] = mer_accuracy_scores
                    results_df['Lev_raw'] = lev_raw_scores
                    results_df['Lev_ratio'] = lev_ratio_scores
                    
                    # Display results
                    st.subheader("Results:")
                    st.dataframe(results_df)
                    
                    # Create download link with original filename + _results + timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    original_name = os.path.splitext(uploaded_file.name)[0]  # Remove .csv extension
                    filename = f"{original_name}_results_{timestamp}.csv"
                    
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="Download Results CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
with tab3:
    st.subheader("About WER/MER Calculator")
    
    st.markdown("<h3 style='color: #1f77b4;'>What is this tool?</h3>", unsafe_allow_html=True)
    st.markdown("""
    This calculator compares reference text (ground truth) with hypothesis text (what was actually transcribed). 
    
    **Common Applications:**
    - **Language Learning Assessment**: Scoring elicited imitation and written dictation tasks for ESL learners
    - **Speech Recognition Evaluation**: Measuring ASR system accuracy
    - **Transcription Quality Control**: Evaluating human or automated transcription services
    """)
    
    st.markdown("<h3 style='color: #1f77b4;'>Key Metrics Explained</h3>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: #ff7f0e;'>Word-Level Metrics:</h4>", unsafe_allow_html=True)
    st.markdown("""
    **Match Error Rate (MER) - RECOMMENDED**
    - **Formula**: MER = (S + D + I) / (S + D + H + I)
    - More appropriate metric than WER (Morris et al., 2004)
    - Includes insertions in denominator, preventing overly punitive measurements
    - **Cannot exceed 1.0**, making it more interpretable
    - 0.0 = Perfect match, 1.0 = Complete mismatch
    
    **Word Error Rate (WER) - Traditional**
    - **Formula**: WER = (S + D + I) / (S + D + H)
    - Included for comparison with traditional literature
    - **Can exceed 1.0** when many insertions occur, making interpretation difficult
    - Insertions negate hits, leading to overly punitive scores
    
    **Key Difference**: MER handles insertions more appropriately by including them in the denominator, while WER allows insertions to inflate the error rate beyond 100%.
    """)
    
    st.markdown("<h4 style='color: #ff7f0e;'>Character-Level Metrics:</h4>", unsafe_allow_html=True)
    st.markdown("""
    **Levenshtein Distance**
    - Character-level edit distance between texts
    - Raw number: total character insertions/deletions/substitutions needed
    - Ratio: similarity score from 0.0 (completely different) to 1.0 (identical)
    - Useful for measuring fine-grained differences in spelling/transcription
    
    *Where: S=Substitutions, D=Deletions, I=Insertions, H=Hits (correct matches)*
    """)
    
    st.markdown("<h3 style='color: #1f77b4;'>Features</h3>", unsafe_allow_html=True)
    st.markdown("""
    **Single Calculation Tab:**
    - Compare two text strings
    - Visual alignment showing exactly where errors occur
    - Real-time error type identification (substitutions, insertions, deletions)
    
    **Batch Processing Tab:**
    - Process CSV files with multiple text pairs
    - Retains all original columns
    - Adds calculated metrics columns including MER_accuracy percentage
    - Exports timestamped results file
    """)
    
    st.markdown("<h3 style='color: #1f77b4;'>Use Cases</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **ESL Assessment**: Scoring elicited imitation and dictation tasks
    - **Language Acquisition Research**: Measuring learner progress over time
    - **ASR Evaluation**: Measuring speech recognition system performance
    - **Quality Control**: Evaluating transcription services or student responses
    """)
    
    st.markdown("<h3 style='color: #1f77b4;'>Technical Notes</h3>", unsafe_allow_html=True)
    st.markdown("""
    - Uses the `jiwer` library for WER/MER calculations
    - Uses the `Levenshtein` library for Levenshtein calculations
    - Handles empty/missing data gracefully in batch processing
    - **All processing happens in your browser - no data is stored**
    """)
    
    st.markdown("<h3 style='color: #1f77b4;'>Reference</h3>", unsafe_allow_html=True)
    st.markdown("""
    Morris, A. C., Maier, V., & Green, P. (2004). From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition. *Proceedings of Interspeech 2004*.
    """)
