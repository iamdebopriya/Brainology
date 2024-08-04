import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model_path = 'brain_tumor_ann_model.h5'
model = tf.keras.models.load_model(model_path)

tumor_advice = {
    "tumor": {
        'medications': (
            "Treatment options for brain tumors may vary depending on the type and stage of the tumor. Common treatments include:\n\n"
            "- **Surgery:** To remove as much of the tumor as possible.\n"
            "- **Chemotherapy:** Uses drugs to kill or stop the growth of cancer cells.\n"
            "- **Radiation Therapy:** Uses high-energy rays to target and destroy cancer cells.\n"
            "- **Targeted Therapy:** Uses drugs that specifically target cancer cells without affecting normal cells.\n"
            "- **Immunotherapy:** Boosts the body's immune system to fight cancer.\n\n"
            "Always consult with a specialist to determine the most appropriate treatment plan based on individual circumstances."
        ),
        'doctors': (
            "The type of specialist you may need to consult can vary based on the type and stage of the tumor:\n\n"
            "- **Neurosurgeon:** A doctor specializing in surgical treatment of brain tumors.\n"
            "- **Oncologist:** A doctor who specializes in diagnosing and treating cancer.\n"
            "- **Neuro-oncologist:** A doctor specializing in brain and spinal cord tumors.\n"
            "- **Radiation Oncologist:** A doctor who uses radiation therapy to treat tumors.\n"
            "- **Medical Oncologist:** A doctor who treats cancer with medication, including chemotherapy, hormonal therapy, and targeted therapy."
        ),
        'diets': (
            "Diet plays a crucial role in supporting overall health and recovery. Recommendations include:\n\n"
            "- **High-Protein Foods:** Such as lean meats, eggs, dairy products, and legumes to help with tissue repair and recovery.\n"
            "- **Fruits and Vegetables:** Rich in vitamins, minerals, and antioxidants to support immune function.\n"
            "- **Healthy Fats:** Include sources of omega-3 fatty acids like fish, nuts, and seeds.\n"
            "- **Hydration:** Ensure adequate fluid intake to stay hydrated.\n"
            "- **Limit Sugars and Processed Foods:** Reduce intake of high-sugar and high-fat foods, which can contribute to inflammation and overall poor health.\n\n"
            "Consult a nutritionist for a personalized diet plan tailored to specific needs and health conditions."
        ),
        'types_of_tumors': (
            "Different types of brain tumors require different treatment approaches and have varying prognoses. Here are some common types:\n\n"
            "- **Gliomas:** Tumors that arise from glial cells. They include:\n  - **Astrocytomas:** Tumors originating from astrocytes.\n  - **Oligodendrogliomas:** Tumors originating from oligodendrocytes.\n  - **Ependymomas:** Tumors originating from ependymal cells.\n\n"
            "- **Meningiomas:** Tumors that form in the meninges, the protective layers surrounding the brain and spinal cord.\n\n"
            "- **Pituitary Tumors:** Tumors that occur in the pituitary gland, affecting hormone levels and function.\n\n"
            "Each type may require a specific treatment approach and has different potential outcomes."
        )
    },
    "normal": {
        'medications': 'No specific medications for a normal brain MRI.',
        'doctors': 'Consult a general physician or neurologist for routine check-ups.',
        'diets': 'Maintain a balanced diet for overall health.'
    }
}

def predict_tumor_type(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Make it a batch of 1
    img_array /= 255.0  # Scale the image

    # Predict the class
    predictions = model.predict(img_array)
    class_indices = {'normal': 0, 'tumor': 1}
    class_labels = list(class_indices.keys())

    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class, predictions[0]

def get_advice(tumor_type):
    advice = tumor_advice.get(tumor_type, {
        'medications': 'No advice available.',
        'doctors': 'No advice available.',
        'diets': 'No advice available.',
        'types_of_tumors': 'No advice available.'
    })
    return advice

st.title('Brainology: Brain Tumor Prediction and Advice')

uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.image(temp_file_path, caption='Uploaded Image.', use_column_width=True)

    # Predict the tumor type
    tumor_type, raw_predictions = predict_tumor_type(temp_file_path)
    st.subheader(f'Predicted Tumor Type: {tumor_type}')

    # Get advice
    advice = get_advice(tumor_type)
    st.subheader('Advice:')
    st.write(f"**Medications:** {advice['medications']}")
    st.write(f"**Doctors:** {advice['doctors']}")
    st.write(f"**Diets:** {advice['diets']}")
    if tumor_type == 'tumor':
        st.write(f"**Types of Tumors:** {advice['types_of_tumors']}")

    # Plotting probabilities
    st.subheader('Prediction Probabilities:')
    fig, ax = plt.subplots(figsize=(10, 6))
    class_labels = ['Normal', 'Tumor']
    sns.barplot(x=class_labels, y=raw_predictions, ax=ax)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Tumor Type')
    ax.set_title('Probability Distribution of Tumor Types')
    st.pyplot(fig)

    # Pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(raw_predictions, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax_pie.set_title('Probability Distribution of Tumor Types')
    st.pyplot(fig_pie)

    # Remove the temporary file
    os.remove(temp_file_path)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #98FF98, #a56dff,#FFD700); 
    }
    .stTitle {
        color: #6a0dad; /* Violet color for titles */
        font-size: 2em; /* Title size */
    }
    .stText {
        color: #6a0dad; /* Violet color for text */
    }
    .stButton > button {
        background-color: #6a0dad; /* Violet background for buttons */
        color: #fff; /* White text color */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #4b0082; /* Darker violet for button hover */
    }
    </style>
""", unsafe_allow_html=True)
