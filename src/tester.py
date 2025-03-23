import gymnasium as gym
import torch
from src.algorithm.SAC import SAC
from src.networks.actor import ActorNetwork

# Path where to save policy model
POLICY_DIR = '/home/flavio/Scrivania/Soft-Actor-Critic-implementation/trained_models/policy.pth'

def tester():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env = gym.make('Pendulum-v1', render_mode = 'human')

    policy = ActorNetwork(state_dim=env.observation_space.shape, max_actions_values=env.action_space.high, device=device)
    policy.load_state_dict(torch.load(POLICY_DIR, map_location=device, weights_only=True))
    
    policy.eval()
    policy.to(device)
    
    state, _ = env.reset()
    with torch.no_grad():
        while True:
                state = torch.tensor(state, dtype=torch.float).to(device)
                env.render()
                action,_ = policy(state)
                action = action.to('cpu').detach().numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
            
    