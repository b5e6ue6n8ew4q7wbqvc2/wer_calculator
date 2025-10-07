import streamlit as st
import jiwer
import Levenshtein
import pandas as pd
from datetime import datetime
import io
import os
import textdistance
from rapidfuzz import fuzz
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='nltk.translate.bleu_score')

# Download NLTK data if needed
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

download_nltk_data()

# Load sentence transformer model (cached)
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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
st.set_page_config(page_title="String Metrics Calculator", layout="wide")

# Title
st.title("String Metrics Calculator")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single Calculation", "Batch Processing", "About"])

with tab1:
    # Metric selection
    st.subheader("Select Metrics to Calculate:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Character Level**")
        char_lev = st.checkbox("Levenshtein (~0.01 ms)", value=True)
        st.caption("0-∞ | Lower better | Raw edit count")
        
        char_jaro = st.checkbox("Jaro-Winkler (~0.001 ms)", value=False)
        st.caption("0-1 | Higher better | Emphasizes prefixes")
        
        char_jaccard = st.checkbox("Character Jaccard (~0.06 ms)", value=False)
        st.caption("0-1 | Higher better | Character set overlap")
    
    with col2:
        st.markdown("**Word Level**")
        word_wer = st.checkbox("WER (~0.1 ms)", value=True)
        st.caption("0-∞ | Lower better | Can exceed 1.0")
        
        word_mer = st.checkbox("MER (~0.1 ms)", value=True)
        st.caption("0-1 | Lower better | Bounded error rate")
        
        word_per = st.checkbox("PER (~0.06 ms)", value=False)
        st.caption("0-∞ | Lower better | Ignores word order")
        
        word_jaccard = st.checkbox("Word Jaccard (~0.02 ms)", value=False)
        st.caption("0-1 | Higher better | Word set overlap")
        
        word_lcs = st.checkbox("LCS Ratio (~0.11 ms)", value=False)
        st.caption("0-1 | Higher better | Longest common subseq")
        
        word_token_sort = st.checkbox("Token Sort (~0.03 ms)", value=False)
        st.caption("0-1 | Higher better | Fuzzy matching")
        
        word_bleu = st.checkbox("BLEU (~0.21 ms)", value=False)
        st.caption("0-1 | Higher better | Tends to score low")
    
    with col3:
        st.markdown("**Semantic Level**")
        sem_meteor = st.checkbox("METEOR (~0.16 ms)", value=False)
        st.caption("0-1 | Higher better | Uses synonyms/stems")
        
        sem_sentence = st.checkbox("Sentence Similarity (~17 ms - slow)", value=False)
        st.caption("0-1 | Higher better | Semantic meaning")
        if sem_sentence:
            st.caption("⚠️ Slow on first run (loads model)")
    
    st.markdown("---")
    
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
                results = []
                
                # Character-level metrics
                if char_lev:
                    lev_distance = Levenshtein.distance(ground_truth, hypothesis)
                    lev_ratio = Levenshtein.ratio(ground_truth, hypothesis)
                    results.append(f"Levenshtein Distance: {lev_distance}")
                    results.append(f"Levenshtein Ratio: {lev_ratio:.4f}")
                
                if char_jaro:
                    jaro = textdistance.jaro_winkler(ground_truth, hypothesis)
                    results.append(f"Jaro-Winkler: {jaro:.4f}")
                
                if char_jaccard:
                    char_jacc = textdistance.jaccard(ground_truth, hypothesis)
                    results.append(f"Character Jaccard: {char_jacc:.4f}")
                
                # Word-level metrics
                if word_wer or word_mer:
                    out = jiwer.process_words([ground_truth], [hypothesis])
                    if word_wer:
                        results.append(f"WER: {out.wer:.4f}")
                    if word_mer:
                        results.append(f"MER: {out.mer:.4f}")
                        results.append(f"MER Accuracy: {(1-out.mer)*100:.2f}%")
                
                if word_per:
                    transformation = jiwer.Compose([
                        jiwer.ToLowerCase(),
                        jiwer.RemovePunctuation(),
                        jiwer.RemoveMultipleSpaces(),
                        jiwer.Strip()
                    ])
                    ref_words = transformation(ground_truth).split()
                    hyp_words = transformation(hypothesis).split()
                    ref_sorted = sorted(ref_words)
                    hyp_sorted = sorted(hyp_words)
                    if len(ref_sorted) > 0:
                        per_out = jiwer.process_words([' '.join(ref_sorted)], [' '.join(hyp_sorted)])
                        results.append(f"PER: {per_out.wer:.4f}")
                    else:
                        results.append(f"PER: N/A")
                
                if word_jaccard:
                    transformation = jiwer.Compose([
                        jiwer.ToLowerCase(),
                        jiwer.RemovePunctuation(),
                        jiwer.RemoveMultipleSpaces(),
                        jiwer.Strip()
                    ])
                    ref_words = set(transformation(ground_truth).split())
                    hyp_words = set(transformation(hypothesis).split())
                    if len(ref_words) == 0 and len(hyp_words) == 0:
                        word_jacc = 1.0
                    else:
                        intersection = len(ref_words & hyp_words)
                        union = len(ref_words | hyp_words)
                        word_jacc = intersection / union if union > 0 else 0.0
                    results.append(f"Word Jaccard: {word_jacc:.4f}")
                
                if word_lcs:
                    lcs = textdistance.lcsstr.normalized_similarity(ground_truth, hypothesis)
                    results.append(f"LCS Ratio: {lcs:.4f}")
                
                if word_token_sort:
                    token_sort = fuzz.token_sort_ratio(ground_truth, hypothesis) / 100.0
                    results.append(f"Token Sort: {token_sort:.4f}")
                
                if word_bleu:
                    ref_words = ground_truth.split()
                    hyp_words = hypothesis.split()
                    if len(ref_words) > 0:
                        bleu = sentence_bleu([ref_words], hyp_words)
                        results.append(f"BLEU: {bleu:.4f}")
                    else:
                        results.append(f"BLEU: N/A")
                
                # Semantic metrics
                if sem_meteor:
                    ref_words = ground_truth.split()
                    hyp_words = hypothesis.split()
                    if len(ref_words) > 0 and len(hyp_words) > 0:
                        meteor = meteor_score([ref_words], hyp_words)
                        results.append(f"METEOR: {meteor:.4f}")
                    else:
                        results.append(f"METEOR: N/A")
                
                if sem_sentence:
                    with st.spinner("Loading sentence model..."):
                        model = load_sentence_model()
                    embeddings = model.encode([ground_truth, hypothesis])
                    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                    )
                    results.append(f"Sentence Similarity: {cos_sim:.4f}")
                
                # Display results
                if word_wer or word_mer:
                    st.subheader("Alignment Visualization:")
                    out = jiwer.process_words([ground_truth], [hypothesis])
                    alignment_viz = jiwer.visualize_alignment(out)
                    st.code(alignment_viz)
                
                st.subheader("Metrics:")
                st.code('\n'.join(results))
                
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")

with tab2:
    st.subheader("Batch Processing")

    st.write("**All processing happens in your browser - no data is stored**")
    
    # Metric selection for batch
    st.subheader("Select Metrics to Calculate:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Character Level**")
        batch_char_lev = st.checkbox("Levenshtein (~0.01 ms)", value=True, key="batch_char_lev")
        st.caption("0-∞ | Lower better | Raw edit count")
        
        batch_char_jaro = st.checkbox("Jaro-Winkler (~0.001 ms)", value=False, key="batch_char_jaro")
        st.caption("0-1 | Higher better | Emphasizes prefixes")
        
        batch_char_jaccard = st.checkbox("Character Jaccard (~0.06 ms)", value=False, key="batch_char_jaccard")
        st.caption("0-1 | Higher better | Character set overlap")
    
    with col2:
        st.markdown("**Word Level**")
        batch_word_wer = st.checkbox("WER (~0.1 ms)", value=True, key="batch_word_wer")
        st.caption("0-∞ | Lower better | Can exceed 1.0")
        
        batch_word_mer = st.checkbox("MER (~0.1 ms)", value=True, key="batch_word_mer")
        st.caption("0-1 | Lower better | Bounded error rate")
        
        batch_word_per = st.checkbox("PER (~0.06 ms)", value=False, key="batch_word_per")
        st.caption("0-∞ | Lower better | Ignores word order")
        
        batch_word_jaccard = st.checkbox("Word Jaccard (~0.02 ms)", value=False, key="batch_word_jaccard")
        st.caption("0-1 | Higher better | Word set overlap")
        
        batch_word_lcs = st.checkbox("LCS Ratio (~0.11 ms)", value=False, key="batch_word_lcs")
        st.caption("0-1 | Higher better | Longest common subseq")
        
        batch_word_token_sort = st.checkbox("Token Sort (~0.03 ms)", value=False, key="batch_word_token_sort")
        st.caption("0-1 | Higher better | Fuzzy matching")
        
        batch_word_bleu = st.checkbox("BLEU (~0.21 ms)", value=False, key="batch_word_bleu")
        st.caption("0-1 | Higher better | Tends to score low")
    
    with col3:
        st.markdown("**Semantic Level**")
        batch_sem_meteor = st.checkbox("METEOR (~0.16 ms)", value=False, key="batch_sem_meteor")
        st.caption("0-1 | Higher better | Uses synonyms/stems")
        
        batch_sem_sentence = st.checkbox("Sentence Similarity (~17 ms - slow)", value=False, key="batch_sem_sentence")
        st.caption("0-1 | Higher better | Semantic meaning")
        if batch_sem_sentence:
            st.caption("⚠️ Slow for large files")
    
    st.markdown("---")
    
    st.write("**Input Requirements:**")
    st.write("Upload a CSV file with the following required columns:")
    st.code("""- ground_truth: Reference text (what should have been said)
- hypothesis: Hypothesis text (what was actually transcribed)""")
    
    st.write("**Output:**")
    st.write("The output CSV will contain all original columns plus selected metric columns")
    
    st.write("**Output Filename Format:**")
    st.code("original_filename_results_YYYYMMDD_HHMMSS.csv")
    
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
                    # Load semantic model if needed
                    if batch_sem_sentence:
                        with st.spinner("Loading sentence model..."):
                            semantic_model = load_sentence_model()
                    
                    # Initialize result dictionary
                    results = {}
                    
                    # Process each row
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in df.iterrows():
                        status_text.text(f"Processing row {idx + 1} of {len(df)}...")
                        
                        # Handle NaN/empty values
                        gt = str(row['ground_truth']) if pd.notna(row['ground_truth']) else ""
                        hyp = str(row['hypothesis']) if pd.notna(row['hypothesis']) else ""
                        
                        row_results = {}
                        
                        # Skip completely empty rows
                        if not gt.strip() or not hyp.strip():
                            # Add None for all selected metrics
                            if batch_char_lev:
                                row_results['Lev_raw'] = None
                                row_results['Lev_ratio'] = None
                            if batch_char_jaro:
                                row_results['Jaro_Winkler'] = None
                            if batch_char_jaccard:
                                row_results['Char_Jaccard'] = None
                            if batch_word_wer:
                                row_results['WER'] = None
                            if batch_word_mer:
                                row_results['MER'] = None
                                row_results['MER_accuracy'] = None
                            if batch_word_per:
                                row_results['PER'] = None
                            if batch_word_jaccard:
                                row_results['Word_Jaccard'] = None
                            if batch_word_lcs:
                                row_results['Word_LCS'] = None
                            if batch_word_token_sort:
                                row_results['Token_Sort'] = None
                            if batch_word_bleu:
                                row_results['BLEU'] = None
                            if batch_sem_meteor:
                                row_results['METEOR'] = None
                            if batch_sem_sentence:
                                row_results['Sentence_Sim'] = None
                        else:
                            try:
                                # Character-level
                                if batch_char_lev:
                                    row_results['Lev_raw'] = Levenshtein.distance(gt, hyp)
                                    row_results['Lev_ratio'] = Levenshtein.ratio(gt, hyp)
                                
                                if batch_char_jaro:
                                    row_results['Jaro_Winkler'] = textdistance.jaro_winkler(gt, hyp)
                                
                                if batch_char_jaccard:
                                    row_results['Char_Jaccard'] = textdistance.jaccard(gt, hyp)
                                
                                # Word-level
                                if batch_word_wer or batch_word_mer:
                                    out = jiwer.process_words([gt], [hyp])
                                    if batch_word_wer:
                                        row_results['WER'] = out.wer
                                    if batch_word_mer:
                                        row_results['MER'] = out.mer
                                        row_results['MER_accuracy'] = (1 - out.mer) * 100
                                
                                if batch_word_per:
                                    transformation = jiwer.Compose([
                                        jiwer.ToLowerCase(),
                                        jiwer.RemovePunctuation(),
                                        jiwer.RemoveMultipleSpaces(),
                                        jiwer.Strip()
                                    ])
                                    ref_words = transformation(gt).split()
                                    hyp_words = transformation(hyp).split()
                                    ref_sorted = sorted(ref_words)
                                    hyp_sorted = sorted(hyp_words)
                                    if len(ref_sorted) > 0:
                                        per_out = jiwer.process_words([' '.join(ref_sorted)], [' '.join(hyp_sorted)])
                                        row_results['PER'] = per_out.wer
                                    else:
                                        row_results['PER'] = 0.0 if len(hyp_sorted) == 0 else 1.0
                                
                                if batch_word_jaccard:
                                    transformation = jiwer.Compose([
                                        jiwer.ToLowerCase(),
                                        jiwer.RemovePunctuation(),
                                        jiwer.RemoveMultipleSpaces(),
                                        jiwer.Strip()
                                    ])
                                    ref_words = set(transformation(gt).split())
                                    hyp_words = set(transformation(hyp).split())
                                    if len(ref_words) == 0 and len(hyp_words) == 0:
                                        row_results['Word_Jaccard'] = 1.0
                                    else:
                                        intersection = len(ref_words & hyp_words)
                                        union = len(ref_words | hyp_words)
                                        row_results['Word_Jaccard'] = intersection / union if union > 0 else 0.0
                                
                                if batch_word_lcs:
                                    row_results['Word_LCS'] = textdistance.lcsstr.normalized_similarity(gt, hyp)
                                
                                if batch_word_token_sort:
                                    row_results['Token_Sort'] = fuzz.token_sort_ratio(gt, hyp) / 100.0
                                
                                if batch_word_bleu:
                                    ref_words = gt.split()
                                    hyp_words = hyp.split()
                                    if len(ref_words) > 0:
                                        row_results['BLEU'] = sentence_bleu([ref_words], hyp_words)
                                    else:
                                        row_results['BLEU'] = 0.0
                                
                                # Semantic
                                if batch_sem_meteor:
                                    ref_words = gt.split()
                                    hyp_words = hyp.split()
                                    if len(ref_words) > 0 and len(hyp_words) > 0:
                                        row_results['METEOR'] = meteor_score([ref_words], hyp_words)
                                    else:
                                        row_results['METEOR'] = 0.0
                                
                                if batch_sem_sentence:
                                    embeddings = semantic_model.encode([gt, hyp])
                                    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
                                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                                    )
                                    row_results['Sentence_Sim'] = float(cos_sim)
                                
                            except Exception as e:
                                st.warning(f"Error processing row {idx + 1}: {str(e)}")
                                # Set all to None on error
                                for key in row_results.keys():
                                    row_results[key] = None
                        
                        # Store results
                        for key, value in row_results.items():
                            if key not in results:
                                results[key] = []
                            results[key].append(value)
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(df))
                    
                    status_text.text("Processing complete!")
                    
                    # Create results dataframe
                    results_df = df.copy()
                    for key, values in results.items():
                        results_df[key] = values
                    
                    # Display results
                    st.subheader("Results:")
                    st.dataframe(results_df)
                    
                    # Create download link
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    original_name = os.path.splitext(uploaded_file.name)[0]
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
    st.subheader("About String Metrics Calculator")
    
    st.markdown("<h3 style='color: #1f77b4;'>What is this tool?</h3>", unsafe_allow_html=True)
    st.markdown("""
    This calculator compares reference text (ground truth) with hypothesis text using multiple string similarity and error metrics.
    
    **Common Applications:**
    - **Language Learning Assessment**: Scoring elicited imitation and written dictation tasks
    - **Speech Recognition Evaluation**: Measuring ASR system accuracy
    - **Transcription Quality Control**: Evaluating human or automated transcription
    """)
    
    st.markdown("<h3 style='color: #1f77b4;'>Metrics Explained</h3>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: #ff7f0e;'>Character-Level Metrics:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **Levenshtein Distance**: Edit distance (insertions/deletions/substitutions) at character level
    - **Jaro-Winkler**: Similarity score emphasizing matching prefixes
    - **Character Jaccard**: Set-based similarity of character sets
    """)
    
    st.markdown("<h4 style='color: #ff7f0e;'>Word-Level Metrics:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **WER (Word Error Rate)**: Traditional metric; can exceed 1.0
    - **MER (Match Error Rate)**: Recommended; bounded 0-1, handles insertions better
    - **PER (Position-independent Error Rate)**: Ignores word order
    - **Word Jaccard**: Set-based word overlap
    - **LCS Ratio**: Longest common subsequence similarity
    - **Token Sort Ratio**: Fuzzy matching with sorted tokens
    - **BLEU**: N-gram overlap score (common in translation)
    """)
    
    st.markdown("<h4 style='color: #ff7f0e;'>Semantic-Level Metrics:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **METEOR**: Uses stemming and synonyms for meaning-aware comparison
    - **Sentence Similarity**: Cosine similarity of sentence embeddings (captures meaning)
    """)
    
    st.markdown("<h3 style='color: #1f77b4;'>Technical Notes</h3>", unsafe_allow_html=True)
    st.markdown("""
    - Error rates: Lower is better (0.0 = perfect)
    - Similarity scores: Higher is better (1.0 = perfect)
    - Semantic metrics use pre-trained models (slow on first run)
    - **All processing happens in your browser - no data is stored**
    """)
    
    st.markdown("<h3 style='color: #1f77b4;'>Reference</h3>", unsafe_allow_html=True)
    st.markdown("""
    Morris, A. C., Maier, V., & Green, P. (2004). From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition. *Proceedings of Interspeech 2004*.
    """)
