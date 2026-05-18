# DL-Project: Brain Tumor Classification

A robust Deep Learning pipeline for detecting and classifying brain tumors from MRI scans. This project encompasses the entire machine learning lifecycle: from exploratory data analysis and custom Convolutional Neural Network (CNN) architecture design, to model training, evaluation, and a fully interactive web dashboard deployment.

## Project Architecture

This repository is structured to provide both research depth and production-ready applications:

- **`brain_tumor_cnn.py`**: The core deep learning module. Contains the PyTorch/Keras Convolutional Neural Network architecture designed specifically for medical image feature extraction.
- **`app.py`**: A dynamic Streamlit web application. This provides a user-friendly GUI allowing users to upload MRI scans, adjust classification thresholds, and view prediction confidences in real-time.
- **`Project1.ipynb`**: A comprehensive Jupyter Notebook detailing the data science workflow. Includes data cleaning, augmentation strategies, model training history, and loss/accuracy visualizations.
- **`dl_utils.py`**: A suite of helper functions for image preprocessing (resizing, normalization, grayscale conversion) and post-processing evaluation metrics.
- **`brain_tumor_model.h5`**: The finalized, pre-trained neural network weights. (Note: Due to its 309MB size, this file is securely tracked using Git Large File Storage).

## Installation and Local Setup

To run this project on your local machine, ensure you have Python 3.8 to 3.11 installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Raheeqi7/DL-Project.git
   cd DL-Project
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Web Dashboard:**
   ```bash
   streamlit run app.py
   ```

## Cloud Deployment Challenges & Solutions

Deploying a complex deep learning computer vision model to cloud servers (like Streamlit Community Cloud) presented several unique challenges that were resolved during development:

1. **Large Model File Size Restrictions (Git LFS)**
   - *Challenge*: The trained model file (`brain_tumor_model.h5`) is 309MB, which strictly exceeds GitHub's standard 100MB file limit. Attempting to push this normally resulted in rejected commits.
   - *Solution*: Implemented **Git Large File Storage (Git LFS)** to specifically track `*.h5` files, allowing the massive model to securely sync with the repository without breaking GitHub limits.

2. **Python Version Compatibility with TensorFlow**
   - *Challenge*: Streamlit Community Cloud defaults to Python 3.14 for new deployments. However, Google's TensorFlow library does not yet have pre-compiled binaries (wheels) for Python 3.14, causing the server installation to crash instantly with `unsatisfiable requirements`.
   - *Solution*: Modified the Streamlit Cloud "Advanced Settings" to specifically force a downgrade to **Python 3.11**, and loosened the `tensorflow>=2.15.0` constraint in `requirements.txt` to allow the package manager to natively resolve the best compatible version.

3. **Headless Server OpenCV GUI Conflicts**
   - *Challenge*: The standard `opencv-python` library assumes it is running on a desktop machine and attempts to link to native graphics drivers (`libGL.so.1`). Cloud servers lack these drivers, triggering an `ImportError: libGL.so.1: cannot open shared object file` crash.
   - *Solution*: Replaced the standard OpenCV package with **`opencv-python-headless`** in `requirements.txt`. This lightweight version strips out the unnecessary desktop GUI dependencies, allowing the image preprocessing pipeline to run flawlessly in a pure server environment.
