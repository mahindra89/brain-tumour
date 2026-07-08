from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "Brain-Tumor-Test-Images"
MODEL_PATH = ROOT / "models" / "bt_resnet50_model.pt"
HERO_IMAGE = ROOT / "static" / "b.jpg"

CLASSES = ["No tumor", "Meningioma", "Glioma", "Pituitary"]

PIPELINE = pd.DataFrame(
    [
        {
            "Stage": "1. MRI input",
            "What happens": "A brain MRI image is uploaded or selected from the sample set.",
            "Why it matters": "The model expects consistent image inputs before inference.",
        },
        {
            "Stage": "2. Preprocessing",
            "What happens": "Images are resized and converted into tensors for the CNN.",
            "Why it matters": "Standardized input size makes predictions reproducible.",
        },
        {
            "Stage": "3. Feature extraction",
            "What happens": "A ResNet-50 backbone learns visual patterns from MRI scans.",
            "Why it matters": "Convolutional features capture texture, boundary, and intensity cues.",
        },
        {
            "Stage": "4. Classification",
            "What happens": "The final classifier maps features to one of four classes.",
            "Why it matters": "The output gives a model prediction category for review.",
        },
        {
            "Stage": "5. Review",
            "What happens": "Predictions should be interpreted as educational output only.",
            "Why it matters": "Medical imaging models require clinical validation before real use.",
        },
    ]
)


def load_image(path_or_file):
    image = Image.open(path_or_file).convert("RGB")
    return ImageOps.exif_transpose(image)


def image_profile(image):
    gray = image.convert("L")
    pixels = list(gray.getdata())
    mean_intensity = sum(pixels) / len(pixels)
    sorted_pixels = sorted(pixels)
    low = sorted_pixels[int(0.05 * (len(sorted_pixels) - 1))]
    high = sorted_pixels[int(0.95 * (len(sorted_pixels) - 1))]
    return {
        "Width": image.width,
        "Height": image.height,
        "Mean intensity": round(mean_intensity, 2),
        "5th percentile": low,
        "95th percentile": high,
    }


st.set_page_config(page_title="Brain Tumor MRI Explorer", layout="wide")

st.title("Brain Tumor MRI Explorer")
st.caption(
    "An interactive portfolio page for a brain MRI classification project using "
    "a ResNet-style deep learning pipeline."
)

st.warning(
    "Educational demo only. This page is not a medical device and must not be "
    "used for diagnosis or treatment decisions."
)

overview_tab, explorer_tab, pipeline_tab, model_tab, learning_tab = st.tabs(
    ["Overview", "Image Explorer", "Pipeline", "Model Status", "Learning"]
)

with overview_tab:
    st.subheader("Project Idea")
    st.markdown(
        """
This project explores how convolutional neural networks can support brain MRI
image classification. The original Flask app was designed to upload an MRI scan
and classify it into tumor-related categories using a PyTorch ResNet-50 model.

For the online portfolio version, the app focuses on a reliable interactive
experience: visitors can inspect sample MRI images, upload their own image for
preprocessing review, understand the model workflow, and see exactly what is
needed to enable live inference.
"""
    )

    cols = st.columns(4)
    cols[0].metric("Task", "MRI classification")
    cols[1].metric("Backbone", "ResNet-50")
    cols[2].metric("Classes", "4")
    cols[3].metric("App type", "Streamlit")

    if HERO_IMAGE.exists():
        st.image(str(HERO_IMAGE), caption="Brain MRI project visual", use_container_width=True)

    st.subheader("Target Classes")
    st.write(", ".join(CLASSES))

with explorer_tab:
    st.subheader("Interactive Image Explorer")
    st.markdown(
        """
Choose one of the included sample MRI images or upload an image to inspect how
it would enter the classification workflow.
"""
    )

    sample_files = sorted(SAMPLE_DIR.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    mode = st.radio("Image source", ["Sample image", "Upload image"], horizontal=True)

    selected_image = None
    selected_name = None

    if mode == "Sample image":
        if sample_files:
            sample_name = st.selectbox("Sample", [p.name for p in sample_files])
            sample_path = SAMPLE_DIR / sample_name
            selected_image = load_image(sample_path)
            selected_name = sample_name
        else:
            st.info("No sample images are available in the repository.")
    else:
        upload = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])
        if upload:
            selected_image = load_image(upload)
            selected_name = upload.name

    if selected_image:
        left, right = st.columns([1, 1])
        with left:
            st.image(selected_image, caption=selected_name, use_container_width=True)
        with right:
            st.markdown("#### Image Profile")
            st.dataframe(
                pd.DataFrame([image_profile(selected_image)]),
                hide_index=True,
                use_container_width=True,
            )

            preview = selected_image.resize((512, 512))
            st.image(preview, caption="512 x 512 preprocessing preview", use_container_width=True)

            if MODEL_PATH.exists():
                st.success("Model artifact found. Live inference can be enabled for this repo.")
            else:
                st.info(
                    "The trained model artifact is not included in the repository, so this "
                    "online version does not generate medical predictions."
                )

with pipeline_tab:
    st.subheader("Classification Workflow")
    st.dataframe(PIPELINE, hide_index=True, use_container_width=True)

    st.markdown(
        """
The original code defines a ResNet-50 model with a custom fully connected
classification head. The intended output classes are no tumor, meningioma,
glioma, and pituitary tumor. The deployed page keeps the workflow visible while
avoiding unsupported predictions when the model file is absent.
"""
    )

with model_tab:
    st.subheader("Model Availability")
    if MODEL_PATH.exists():
        st.success(f"Model file detected: `{MODEL_PATH.name}`")
        st.write("The next step would be wiring this artifact into the Streamlit inference path.")
    else:
        st.warning("The trained model file is not currently included in GitHub.")
        st.markdown(
            """
The legacy Flask app expects this file:

```text
models/bt_resnet50_model.pt
```

Without that file, an online app cannot make real model predictions. The project
therefore uses an interactive explainer and image-inspection workflow until the
model artifact is added or hosted from a reliable download location.
"""
        )

    st.subheader("Deployment Checklist")
    checklist = pd.DataFrame(
        [
            {"Item": "Streamlit app entrypoint", "Status": "Ready", "Notes": "`app.py`"},
            {"Item": "Sample MRI images", "Status": "Ready", "Notes": "Included in repository"},
            {"Item": "Model artifact", "Status": "Missing", "Notes": "`models/bt_resnet50_model.pt`"},
            {"Item": "Medical disclaimer", "Status": "Ready", "Notes": "Shown in app"},
            {"Item": "Cloud dependencies", "Status": "Ready", "Notes": "Minimal Streamlit/Pillow stack"},
        ]
    )
    st.dataframe(checklist, hide_index=True, use_container_width=True)

with learning_tab:
    st.subheader("What This Project Demonstrates")
    st.markdown(
        """
This project demonstrates the structure of a medical-imaging classification
application: image upload, preprocessing, model-backed classification, and a
user-facing web interface.

The strongest engineering lesson is deployment readiness. A model project is
not only the neural network architecture; it also needs a reliable artifact
strategy, clear dependency management, safe user-facing language, and a
transparent explanation of what is and is not available in the deployed app.
"""
    )

    st.markdown(
        """
#### Technical Concepts

- MRI image preprocessing
- ResNet-based transfer learning
- Multi-class image classification
- Streamlit deployment
- Responsible medical AI framing
"""
    )
