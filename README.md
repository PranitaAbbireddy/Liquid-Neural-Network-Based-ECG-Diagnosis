# Liquid-Neural-Network-Based-ECG-Diagnosis

This project implements a Liquid Neural Network (LNN) using PyTorch to classify ECG signals into multiple arrhythmia classes. The model is trained on the MIT-BIH Arrhythmia Database and evaluated with interpretability methods such as Integrated Gradients (IG) and Saliency Maps.

It provides:

- ECG classification with high accuracy (~97%).
- Visualizations of training performance and confusion matrix.
- ROC curves for multi-class classification.
- Model explainability with Captum (IG, Saliency).
- Attribution summaries for different arrhythmia classes.

## Evaluation metrics

- Accuracy
- Confusion Matrix
- Classification Report
- ROC Curve (per class)

### Explainability with Captum:

- Integrated Gradients (IG)
- Saliency Maps
- Forced-class IG visualization

## Dataset

### MIT-BIH Arrhythmia Database

Train & Test CSV files with 187 time steps per ECG beat.
- 5 classes:
  0: Normal
  1: Supraventricular
  2: Ventricular
  3: Fusion
  4: Unknown

### PTB Diagnostic ECG Database (PTBDB)

Used for Normal vs Abnormal visualization & attribution.

## Model
Liquid Neural Network (PyTorch)

    class LiquidECGClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LiquidECGClassifier, self).__init__()
            self.liquid_layer = nn.RNN(input_size, hidden_size, nonlinearity='tanh', batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1, input_size)  # Reshape for RNN input
        x, _ = self.liquid_layer(x)
        return self.fc(x[:, -1, :])  # Last timestep for classification


## Results
### Training Loss
Loss decreases smoothly over epochs.

### Confusion Matrix
Visualizes classification performance per class.

### Classification Report

                    precision    recall  f1-score   support
      Normal            0.98      0.99      0.98     18118
      Supraventricular  0.86      0.62      0.72       556
      Ventricular       0.93      0.90      0.91      1448
      Fusion            0.74      0.59      0.66       162
      Unknown           0.94      0.97      0.96      1608
      
      Overall Accuracy: 96.95%

### ROC Curve
Multi-class ROC curve with AUC per class.

### Interpretability (Captum)
#### Integrated Gradients (IG)

- Shows which time steps contribute most to classification.
- Plotted for each class and for Normal/Abnormal PTBDB samples.
- Supports forced-class attributions for hypothesis testing.

#### Saliency Maps

- Gradient-based explanation of model predictions.
- Compared side-by-side with IG.

## Requirements

Install dependencies:

      pip install torch torchvision torchaudio
      pip install pandas numpy matplotlib seaborn scikit-learn
      pip install shap captum

* Captum: Model Interpretability for PyTorch
