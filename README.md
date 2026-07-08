# Brain Tumor MRI Explorer

This repository contains an interactive Streamlit portfolio page for a brain MRI
classification project. The original application was a Flask app that expected a
trained PyTorch ResNet-50 model artifact for live tumor classification.

The current online version is designed to work reliably on Streamlit Cloud:

- explore included sample MRI images
- upload an image for preprocessing review
- inspect image dimensions and intensity statistics
- understand the intended ResNet-50 classification pipeline
- see what is required to enable real model inference

## Safety Notice

This project is an educational demo only. It is not a medical device and must
not be used for diagnosis, treatment decisions, or clinical screening.

## Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Cloud

Use:

```text
Main file path: app.py
```

The app uses only lightweight dependencies listed in `requirements.txt`.

## Model Artifact

The legacy Flask app expected this file:

```text
models/bt_resnet50_model.pt
```

That trained model file is not currently included in the repository. Because of
that, the deployed app does not claim live tumor predictions. It provides an
interactive image explorer and deployment-ready project explanation instead.

To enable live predictions later:

1. Add `models/bt_resnet50_model.pt`, or host it from a reliable download URL.
2. Add PyTorch and TorchVision dependencies.
3. Load the model in Streamlit with caching.
4. Run inference only with clear educational/medical disclaimers.

## Project Structure

```text
.
|-- app.py                    # Streamlit app for the online project page
|-- legacy_flask_app.py       # Original Flask implementation
|-- Brain-Tumor-Test-Images/  # Sample MRI images used by the app
|-- models/                   # Placeholder for the trained model artifact
|-- static/                   # Original static assets
|-- templates/                # Original Flask templates
|-- requirements.txt          # Streamlit Cloud dependencies
`-- README.md
```

## Intended Model Workflow

The original code defines a ResNet-50 classifier with a custom fully connected
head. The intended output classes are:

- No tumor
- Meningioma
- Glioma
- Pituitary

The expected workflow is:

1. Upload or select an MRI image.
2. Resize and convert the image into a tensor.
3. Pass the tensor through the ResNet-50 feature extractor.
4. Classify the MRI image into one of the target classes.
5. Present the output with a clear medical safety disclaimer.

## Next Improvements

- Add the trained model artifact or a reliable model download path.
- Add a small validation/results section with accuracy, precision, recall, and F1.
- Add example predictions from a fixed sample set.
- Add Grad-CAM or saliency visualizations for explainability.
- Separate model-loading utilities from the UI code once inference is enabled.
