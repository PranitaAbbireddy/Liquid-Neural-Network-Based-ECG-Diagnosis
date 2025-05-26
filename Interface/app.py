from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import pandas as pd
from flask_cors import CORS
import random
import datetime

app = Flask(__name__)
CORS(app)

class LiquidNeuralNetwork(nn.Module):
    def __init__(self):
        super(LiquidNeuralNetwork, self).__init__()
        self.liquid_layer = nn.RNN(input_size=187, hidden_size=512, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        out, _ = self.liquid_layer(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LiquidNeuralNetwork()
model.load_state_dict(torch.load("ecg_model.pt", map_location=torch.device("cpu")))
model.eval()

LABELS = [
    "Normal Beat (N)",
    "Atrial Premature Beat (A)",
    "Ventricular Premature Beat (V)",
    "Myocardial Infarction (MI)",
    "Other Arrhythmias / Fusion Beats / Unknown"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        df = pd.read_csv(file, header=None)
        if df.shape[1] == 188:
            df = df.iloc[:, 1:]
        input_tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        result = {
            "class": predicted_class,
            "label": LABELS[predicted_class],
            "report_id": f"RPT-{random.randint(100000, 999999)}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
