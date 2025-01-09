import torch
from model import SolitaireModel, SolitaireCNNAttModel


class QAgent:
    def __init__(self, model_dict, optimizer, lr, loss_fn, tau=0.005, discount_factor=0.99, device='cpu'):
        self.model = SolitaireCNNAttModel(**model_dict)
        self.target_model = SolitaireCNNAttModel(**model_dict)
        self.tau = tau
        self.discount_factor = discount_factor
        optimizer = optimizer(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def select_action(self, state, policy='softmax', temperature=1, weights=None):
        with torch.no_grad():
            q_values = self.model(state)
        if policy == 'softmax':
            probs = torch.softmax(q_values, -1)
            probs = probs.pow(1/temperature)
            action = torch.multinomial(probs, 1)
        elif policy == 'weights':
            weights = torch.tensor(weights).unsqueeze(0).to(self.device)
            action = torch.multinomial(weights, 1)
        return action, q_values.gather(1, action)

    def calculate_loss(self, q_value, rewards, next_state, value):
        with torch.no_grad():
            next_q_values = self.target_model(next_state)
        target = rewards + self.discount_factor * next_q_values.max(-1).values * value
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        if q_value.ndim == 1:
            q_value = q_value.unsqueeze(-1)
        return self.loss_fn(q_value, target)

    def step(self, sample):
        state, next_state, actions, rewards, value = sample
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        value = value.to(self.device)
        self.optimizer.zero_grad()
        q_values = self.model(state)
        q_action_value = q_values.gather(1, actions.unsqueeze(-1))
        loss = self.calculate_loss(q_action_value, rewards, next_state, value)
        loss.backward()
        self.optimizer.step()
        self.update_weights()
        return loss.item(), q_values

    def update_weights(self):
        target_state_dict = self.target_model.state_dict()
        policy_state_dict = self.model.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * self.tau + \
                target_state_dict[key] * (1 - self.tau)
        self.target_model.load_state_dict(target_state_dict)

    def save_weights(self):
        torch.save(self.model.state_dict(), "model_weights.pth")
        torch.save(self.target_model.state_dict(), "target_model_weights.pth")

    def load_weights(self):
        self.model.load_state_dict(torch.load("model_weights.pth"))
        self.target_model.load_state_dict(torch.load("target_model_weights.pth"))
