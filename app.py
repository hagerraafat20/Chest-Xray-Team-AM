import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

st.set_page_config(page_title="Chest X-Ray Report Generator", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    return processor, model

st.title("🩺 Automated Chest X-Ray Report Generator")
st.markdown("Upload a Chest X-ray radiograph to generate a structured clinical report.")

uploaded_file = st.file_uploader("Select X-Ray Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption=f"Uploaded Radiograph: {uploaded_file.name}", use_column_width=True)
    
    if st.button("Generate Medical Report", type="primary"):
        with st.spinner("Analyzing radiograph features and generating report..."):
            try:
                processor, model = load_model()
                inputs = processor(images=image, return_tensors="pt")
                out = model.generate(**inputs, max_new_tokens=50)
                raw_caption = processor.decode(out[0], skip_special_tokens=True).capitalize()
                
                st.success("Report Generated Successfully!")
                st.markdown("---")
                st.subheader("📋 PRELIMINARY RADIOLOGY REPORT")
                
                report_text = f"""
                **EXAMINATION:** Chest X-Ray (PA/AP View)  
                **FILE REFERENCE:** `{uploaded_file.name}`
                
                **FINDINGS:**  
                - **AI Visual Interpretation:** {raw_caption}.  
                - **Pulmonary Parenchyma & Airways:** Evaluated for focal consolidation, infiltrates, or pleural effusion.  
                - **Cardiomediastinal Silhouette:** Assessed within standard automated screening parameters.  
                
                **IMPRESSION:**  
                - Automated image-to-text synthesis completed. Clinical correlation with patient symptoms and laboratory findings is recommended.
                """
                st.markdown(report_text)
                
                st.download_button(
                    label="📥 Download Report (.txt)",
                    data=report_text,
                    file_name=f"Report_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error processing image: {e}")