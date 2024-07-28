import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save the trained model
def save_model(model, cuboid_dimensions, action_size, file_path):
    model_info = {
        "cuboid_dimensions": cuboid_dimensions,
        "action_size": action_size,
        "state_dict": model.state_dict()
    }
    torch.save(model_info, file_path)

# Load the model for inference
def load_model(file_path):
    model_info = torch.load(file_path, map_location=device)
    cuboid_dimensions = model_info["cuboid_dimensions"]
    action_size = model_info["action_size"]
    state_dict = model_info["state_dict"]
    
    model = DQN(cuboid_dimensions, action_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    return model

# Helper function to get all rotations of an item
def get_rotations(item):
    rotations = [
        (item[0], item[1], item[2]),
        (item[0], item[2], item[1]),
        (item[1], item[0], item[2]),
        (item[1], item[2], item[0]),
        (item[2], item[0], item[1]),
        (item[2], item[1], item[0])
    ]
    return rotations

# Define maximum dimensions and item count for padding
MAX_CUBOID_DIMENSIONS = (15, 15, 15)
MAX_ITEMS = 10

# Environment
class PackingEnvironment:
    def __init__(self, cuboid_dimensions, items):
        self.cuboid_dimensions = cuboid_dimensions
        self.items = items
        self.reset()

    def reset(self):
        self.cuboid = np.zeros(self.cuboid_dimensions)
        self.current_item_index = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.done or self.current_item_index >= len(self.items):
            cuboid_flattened = np.zeros(np.prod(MAX_CUBOID_DIMENSIONS))
            item_dims = np.zeros(3)
        else:
            cuboid_flattened = np.pad(self.cuboid.flatten(), 
                                      (0, np.prod(MAX_CUBOID_DIMENSIONS) - np.prod(self.cuboid_dimensions)), 
                                      mode='constant')
            item_dims = np.pad(np.array(self.items[self.current_item_index]), 
                               (0, 3 - len(self.items[self.current_item_index])), 
                               mode='constant')
        state = np.concatenate((cuboid_flattened, item_dims))
        return state

    def step(self, position_rotation):
        if self.done:
            raise Exception("Environment is done. Reset to start again.")
        
        position, rotation = position_rotation
        item = self.items[self.current_item_index]
        rotated_item = get_rotations(item)[rotation]
        
        if self._can_place_item(position, rotated_item):
            self._place_item(position, rotated_item)
            reward = 1
            self.current_item_index += 1
            if self.current_item_index >= len(self.items):
                self.done = True
        else:
            reward = -0.1  # Slightly negative reward for unsuccessful placement
        
        next_state = self._get_state()
        return next_state, reward, self.done

    def _can_place_item(self, position, item):
        x, y, z = np.unravel_index(position, self.cuboid_dimensions)
        item_x, item_y, item_z = item
        if (x + item_x > self.cuboid_dimensions[0] or
            y + item_y > self.cuboid_dimensions[1] or
            z + item_z > self.cuboid_dimensions[2]):
            return False
        if np.any(self.cuboid[x:x+item_x, y:y+item_y, z:z+item_z]):
            return False
        return True

    def _place_item(self, position, item):
        x, y, z = np.unravel_index(position, self.cuboid_dimensions)
        item_x, item_y, item_z = item
        self.cuboid[x:x+item_x, y:y+item_y, z:z+item_z] = 1

# DQN Model
class DQN(nn.Module):
    def __init__(self, cuboid_dimensions, action_size):
        super(DQN, self).__init__()
        self.cuboid_dimensions = cuboid_dimensions
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Calculate the size after the conv layers
        conv_output_size = 128 * np.prod(cuboid_dimensions)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + 3, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, state):
        cuboid_state = state[:, :-3].view(-1, 1, *self.cuboid_dimensions)
        item_state = state[:, -3:]
        
        cuboid_state = F.relu(self.conv1(cuboid_state))
        cuboid_state = F.relu(self.conv2(cuboid_state))
        cuboid_state = F.relu(self.conv3(cuboid_state))
        
        cuboid_state = cuboid_state.view(cuboid_state.size(0), -1)
        
        combined = torch.cat((cuboid_state, item_state), dim=1)
        combined = self.dropout(F.relu(self.fc1(combined)))
        q_values = self.fc2(combined)
        
        return q_values

# Agent
class DQNAgent:
    def __init__(self, state_size, action_size, cuboid_dimensions, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.cuboid_dimensions = cuboid_dimensions
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = DQN(cuboid_dimensions, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        if self.model.training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(agent, num_episodes, max_items):
    for episode in range(num_episodes):
        cuboid_dimensions = (random.randint(5, 15), random.randint(5, 15), random.randint(5, 15))
        action_size = np.prod(cuboid_dimensions) * 6  # Update action_size based on cuboid dimensions
        items = [(random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)) for _ in range(max_items)]
        env = PackingEnvironment(cuboid_dimensions, items)
        state = env.reset()
        total_reward = 0
        agent.action_size = action_size  # Update agent's action_size based on current cuboid dimensions
        while True:
            action = agent.select_action(state)
            position = action // 6
            rotation = action % 6
            if position >= np.prod(cuboid_dimensions):  # Ensure position is within bounds
                reward = -1
                done = True
                next_state = state
            else:
                next_state, reward, done = env.step((position, rotation))
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.train()
        print(f"Episode {episode+1}, Total Reward: {total_reward}")

def run_inference(env, agent, cuboid_dimensions, items):
    state = env.reset()
    positions = []
    rotations = []
    while not env.done:
        action = agent.select_action(state)
        position = action // 6
        rotation = action % 6
        if position < np.prod(cuboid_dimensions):
            next_state, _, done = env.step((position, rotation))
            state = next_state
            positions.append(position)
            rotations.append(rotation)
        else:
            break

    print("Cuboid dimensions:", cuboid_dimensions)    
    print("Positions:", positions)
    print("Rotations:", rotations)
    
    # Convert positions to 3D coordinates
    coordinates = [np.unravel_index(pos, cuboid_dimensions) for pos in positions]
    print("3D Coordinates:", coordinates)

    # Interpret rotations
    rotated_items = [get_rotations(item)[rot] for item, rot in zip(items, rotations)]
    print("Rotated Items:", rotated_items)    

    return positions, rotations

# Hyperparameters
# Define maximum dimensions and item count for padding
MAX_CUBOID_DIMENSIONS = (15, 15, 15)
MAX_ITEMS = 10

state_size = np.prod(MAX_CUBOID_DIMENSIONS) + 3
action_size = np.prod(MAX_CUBOID_DIMENSIONS) * 6
num_episodes = 50

# Initialize agent and train
agent = DQNAgent(state_size, action_size, MAX_CUBOID_DIMENSIONS)
train_dqn(agent, num_episodes, MAX_ITEMS)
print("Training loop done")

# Save the trained model
save_model(agent.model, MAX_CUBOID_DIMENSIONS, action_size, "trained_model_2.pth")
print("Model saved")

# Define dimensions and items for inference
cuboid_dimensions = (10, 10, 10)  # Example fixed dimensions for inference
items = [(random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)) for _ in range(MAX_ITEMS)]

# Load the model for inference
model_path = "trained_model_2.pth"
loaded_model = load_model(model_path)

inference_agent = DQNAgent(state_size, np.prod(cuboid_dimensions) * 6, cuboid_dimensions)
inference_agent.model = loaded_model

# Create a new environment for inference
new_env = PackingEnvironment(cuboid_dimensions, items)

# Run inference
positions, rotations = run_inference(new_env, inference_agent, cuboid_dimensions, items)

# Print the results
print("Positions:", positions)
print("Rotations:", rotations)
