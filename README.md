# Human-Emotion-Recognition-from-Text-and-Voice-Data

Human Emotion Recognition from Text and Voice Data
This project is an advanced system designed to recognize human emotions from both textual and voice data. It incorporates cutting-edge technologies including deep learning, Natural Language Processing (NLP), and audio processing. The system achieves an accuracy of 87% in emotion recognition, contributing significantly to fields like human-computer interaction, customer service, and mental health support systems.

Features
Text data processing using BERT for contextualized embeddings.
Voice data processing using Mel-Frequency Cepstral Coefficients (MFCCs), CNNs, and RNNs for sequence modeling.
A unified model combining features from both text and audio data using LSTMs and a Multi-Layer Perceptron (MLP).
Evaluation metrics including precision, F1-score, and accuracy.
Installation
Ensure you have Python 3.x installed on your system. Then, install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Project Structure
text_processing/: Contains code for processing text data using BERT.
voice_processing/: Contains code for voice data processing, including feature extraction.
model/: Contains the LSTM and MLP model for emotion recognition.
evaluation/: Contains code for evaluating the model (precision, F1-score, accuracy).
main.py: Main script to run the project.
Usage
To run the project, execute the main.py script:

bash
Copy code
python main.py
Contributing
Contributions to this project are welcome. Please ensure that any pull requests or changes adhere to the existing coding style and structure.

License
This project is licensed under [SPECIFY LICENSE]. For more information, see LICENSE file.

Acknowledgments
Thanks to the developers of the BERT model and Hugging Face Transformers library.
Appreciation for the libraries and technologies used in this project.
