import torch
import torch.nn as nn

class EmotionRecognitionModel(nn.Module):
    def __init__(self, text_feature_dim, audio_feature_dim):
        super(EmotionRecognitionModel, self).__init__()
        # Define the LSTM for text and audio feature combination
        self.lstm = nn.LSTM(input_size=text_feature_dim + audio_feature_dim, hidden_size=128, num_layers=2, batch_first=True)
        # Define the MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # Assuming 8 emotions
        )

    def forward(self, text_feature, audio_feature):
        # Combine text and audio features
        combined_feature = torch.cat((text_feature, audio_feature), dim=1)
        lstm_output, _ = self.lstm(combined_feature)
        # Pass the LSTM output through the MLP
        output = self.mlp(lstm_output[:, -1, :])  # Use the last output of LSTM
        return output
