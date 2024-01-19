import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_mfcc(audio_file):
    # Load the audio file and extract MFCC features
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.T

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        # Define a simple CNN-LSTM architecture
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=2, batch_first=True)

    def forward(self, x):
        # Forward pass through the CNN-LSTM network
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1, 16)  # Reshape for LSTM
        x, _ = self.lstm(x)
        return x
