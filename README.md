# Handwritten Digit Recognizer with CNN and Streamlit

This project is a **Handwritten Digit Recognizer** built using Convolutional Neural Networks (CNN) in Python. It provides a simple interface for digit recognition from images using a trained neural network.

## Features

- **CNN Model**: Trained on the MNIST dataset to classify handwritten digits (0-9).
- **Streamlit UI**: Offers an interactive user interface for digit recognition.

## Repository Structure

- `model/`: Contains the CNN model code and saved weights.
- `streamlit_app.py`: Streamlit interface code for digit recognition.
- `notebooks/`: Jupyter notebooks for data exploration, training, and evaluation.
- `requirements.txt`: Python dependencies.

## Important Note

> **Deployment on Streamlit Cloud is not available for this project.**

Due to compatibility issues between **TensorFlow** and other required libraries, the site cannot currently be deployed on Streamlit Cloud. If you wish to run the Streamlit interface locally, please ensure you have compatible versions of Python and TensorFlow installed.

## Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SaadZaidi1/Digit_Recognizer.git
    cd Digit_Recognizer
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Streamlit app (locally):**
    ```bash
    streamlit run streamlit_app.py
    ```

## Requirements

- Python 3.7+
- TensorFlow (see `requirements.txt` for specific version)
- Streamlit

## License

This project is licensed under the MIT License.

---

**Author:** Saad Zaidi
