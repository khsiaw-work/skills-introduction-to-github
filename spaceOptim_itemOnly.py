import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            return np.zeros(np.prod(self.cuboid_dimensions) + 3)  # Updated to 3 for current item dimensions
        cuboid_flattened = self.cuboid.flatten()
        item_dims = np.array(self.items[self.current_item_index])
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
            reward = 1  # Reward for successfully placing an item
            self.current_item_index += 1
            if self.current_item_index >= len(self.items):
                self.done = True
        else:
            reward = -1  # Penalty for an invalid placement
        
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


# Enhanced DQN Network with CNN
class DQN(nn.Module):
    def __init__(self, cuboid_dimensions, output_dim):
        super(DQN, self).__init__()
        self.cuboid_dimensions = cuboid_dimensions

        # Convolutional layers to process cuboid state
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)

        # Calculate the flattened size after convolutions
        conv_output_size = self._get_conv_output_size(cuboid_dimensions)

        # Fully connected layers to process item dimensions and combined state
        self.fc1 = nn.Linear(conv_output_size + 3, 256)  # Including item dimensions
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim * 6)  # 6 possible rotations

    def _get_conv_output_size(self, shape):
        # Helper function to calculate the size after the convolutional layers
        dummy_input = torch.zeros(1, 1, *shape)
        output = self.conv1(dummy_input)
        output = self.conv2(output)
        output = self.conv3(output)
        return int(np.prod(output.size()))

    def forward(self, x):
        cuboid_state, item_dims = x[:, :-3], x[:, -3:]
        cuboid_state = cuboid_state.view(-1, 1, *self.cuboid_dimensions)

        x = F.relu(self.conv1(cuboid_state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten the output of the convolutional layers
        x = torch.cat((x, item_dims), dim=1)  # Concatenate with item dimensions

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Agent:
    def __init__(self, model, env, lr=0.001, gamma=0.99, epsilon=0.1, batch_size=32, memory_size=1000, training=True):
        self.model = model.to(device)
        self.env = env
        self.training = training
        if self.training:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
            self.memory = deque(maxlen=memory_size)
            self.gamma = gamma
            self.epsilon = epsilon
            self.batch_size = batch_size

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        if self.training and random.random() < self.epsilon:
            position = random.randint(0, np.prod(self.env.cuboid_dimensions) - 1)
            rotation = random.randint(0, 5)
            return (position, rotation)
        with torch.no_grad():
            q_values = self.model(state)
        action = q_values.argmax().item()
        position = action // 6  # Integer division to get the position index
        rotation = action % 6  # Modulus to get the rotation index
        return (position, rotation)

    def store_transition(self, transition):
        if self.training:
            self.memory.append(transition)

    def train(self):
        if not self.training or len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        positions = torch.tensor([a[0] for a in actions], dtype=torch.int64).to(device)
        rotations = torch.tensor([a[1] for a in actions], dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        q_value = q_values[range(self.batch_size), positions * 6 + rotations]
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = F.mse_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def generate_random_items(max_items):
    items = []
    for _ in range(max_items):
        item_dims = tuple(np.random.randint(1, 5, size=3))
        items.append(item_dims)
    return items


def train_dqn(agent, env, num_episodes, max_items):
    for episode in range(num_episodes):
        # Randomly select number of items for this episode
        num_items = np.random.randint(1, max_items + 1)
        env.items = generate_random_items(num_items)
        state = env.reset()
        done = False
        total_reward = 0
        
        # Log the items for this episode
        print(f"Episode {episode + 1}, Training with items: {env.items}")
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition((state, action, reward, next_state, done))
            agent.train()
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


# Save the trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load the trained model
def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# Running inference
def run_inference(env, agent, cuboid_dimensions, items):
    env.items = items
    state = env.reset()
    done = False
    positions = []
    rotations = []
    
    while not done:
        action = agent.select_action(state)
        position, rotation = action
        next_state, reward, done = env.step(action)
        state = next_state
        positions.append(position)
        rotations.append(rotation)

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


# Function to create the vertices for a cuboid
def cuboid_data(o, size=(1, 1, 1)):
    X = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1]])
    X = X * size
    X = X + np.array(o)
    return X



# Plot function
def plot_cuboid(cuboid_dimensions, items, positions, rotations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot cuboid boundaries
    x_dim, y_dim, z_dim = cuboid_dimensions
    ax.set_xlim([0, x_dim])
    ax.set_ylim([0, y_dim])
    ax.set_zlim([0, z_dim])

    # Generate unique colors
    num_items = len(items)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(num_items)]

    for i, (item, position, rotation) in enumerate(zip(items, positions, rotations)):
        rotated_item = get_rotations(item)[rotation]
        x, y, z = np.unravel_index(position, cuboid_dimensions)
        dx, dy, dz = rotated_item

        # Get the vertices for the cuboid
        vertices = cuboid_data((x, y, z), (dx, dy, dz))

        # Define the faces of the cuboid
        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],  # bottom face
            [vertices[j] for j in [4, 5, 6, 7]],  # top face
            [vertices[j] for j in [0, 1, 5, 4]],  # front face
            [vertices[j] for j in [2, 3, 7, 6]],  # back face
            [vertices[j] for j in [1, 2, 6, 5]],  # right face
            [vertices[j] for j in [0, 3, 7, 4]]   # left face
        ]

        # Add the faces as a Poly3DCollection with a unique color
        ax.add_collection3d(Poly3DCollection(faces, facecolors=colors[i], linewidths=1, edgecolors='r', alpha=.25))

        # Add labels to the items
        center = np.mean(vertices, axis=0)
        ax.text(center[0], center[1], center[2], str(i), color='black', fontsize=12, ha='center', va='center')

    plt.show()



# Parameters
cuboid_dimensions = (10, 10, 10)
num_items = 10
items = generate_random_items(num_items)
output_dim = np.prod(cuboid_dimensions)

# Environment and Model
env = PackingEnvironment(cuboid_dimensions, items)
model = DQN(cuboid_dimensions, output_dim)

# Agent
agent = Agent(model, env, training=True)

# Train the model
num_episodes = 100
max_items = 6
train_dqn(agent, env, num_episodes, max_items)

# Save the trained model
save_model(model, "trained_model.pth")

# Load the model for inference
loaded_model = DQN(cuboid_dimensions, output_dim)
loaded_model = load_model(loaded_model, "trained_model.pth")

# Create an agent for inference
inference_agent = Agent(loaded_model, env, training=False)

# Define new dimensions and items for inference
new_dimensions = (10, 10, 10)
new_items = [(3, 2, 2), (1, 3, 3), (2, 1, 3)]
new_env = PackingEnvironment(new_dimensions, new_items)

# Run inference
positions, rotations = run_inference(new_env, inference_agent, new_dimensions, new_items)

# Plot the cuboid with the placed items
plot_cuboid(new_dimensions, new_items, positions, rotations)
