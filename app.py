from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

app = Flask(__name__)

#####################################
# Define the Policy Network (Same as before)
#####################################
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        # Learnable log standard deviation for each action dimension
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_layer(x)
        # Bound the mean to [-5,5] using tanh
        mean = torch.tanh(mean) * 5.0
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)  # shape: [1,5]
        mean, std = self.forward(obs_t)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)  # sum over action dimensions
        return action.detach().cpu().numpy()[0], log_prob

# Instantiate and load your pre-trained model.
# For demonstration, we initialize a new model.
policy = PolicyNetwork()
# In practice, you might load saved weights:
# policy.load_state_dict(torch.load("camera_angle_model.pth", map_location=torch.device('cpu')))
policy.eval()  # set model to evaluation mode

#####################################
# Prediction Endpoint
#####################################
@app.route('/predict', methods=['POST'])
def predict():
    # Expected input JSON structure:
    # {
    #   "bbox_x": float,
    #   "bbox_y": float,
    #   "pitch": float,
    #   "roll": float,
    #   "yaw": float
    # }
    data = request.get_json(force=True)
    try:
        obs = np.array([
            data.get("bbox_x", 50),  # default center
            data.get("bbox_y", 50),
            data.get("pitch", 0),
            data.get("roll", 0),
            data.get("yaw", 0)
        ], dtype=np.float32)
    except Exception as e:
        return jsonify({"error": "Invalid input format", "details": str(e)}), 400

    action, _ = policy.get_action(obs)
    dx, dy = action.tolist()

    # Create a simple feedback text (could be more detailed based on business rules)
    feedback_text = f"Adjust camera by moving dx={dx:.2f} and dy={dy:.2f}."
    
    # You could also return additional information, such as a predicted reward estimate.
    response = {
        "dx": dx,
        "dy": dy,
        "feedback_text": feedback_text,
        "detected_object": "person",  # Example static label; in practice, use YOLO detections.
        "reward_estimate": 0.0         # Placeholder value
    }
    return jsonify(response)

#####################################
# Feedback Endpoint (for collecting user feedback)
#####################################
@app.route('/feedback', methods=['POST'])
def feedback():
    # Expected input JSON structure:
    # {
    #   "session_id": "unique-session-id", (optional)
    #   "user_rating": int  (e.g., rating 1-5)
    # }
    data = request.get_json(force=True)
    user_rating = data.get("user_rating")
    session_id = data.get("session_id", "default_session")

    # For demonstration, we simply print the feedback.
    # In a production setting, you would store this in a database or a replay buffer.
    print(f"Received feedback for session {session_id}: Rating = {user_rating}")
    
    # Acknowledge receipt of feedback.
    return jsonify({"message": "Feedback received", "session_id": session_id})

#####################################
# Run Flask App
#####################################
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
