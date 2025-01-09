from agent import QAgent
from replay_buffer import ReplayBuffer
from solitaire_base import SolitaireGame
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np

np.set_printoptions(suppress=True)

lr = 0.00003
list_legal_moves = []
legal_moves = 0
episodes = int(1e8)
batch_size = 16
train_step = 4
show_step = 1000
step = 0
wins = 0
game = SolitaireGame()
buffer = ReplayBuffer(10000)
losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dict = {'embedding_dims': [4, 8],
              'dense_units': 32,
              'num_layers': 2,
              'nhead': 4,
              'device': device}
# model = SolitaireModel([4, 8], 32, 2, 4)
q_agent = QAgent(model_dict, Adam, lr=lr, loss_fn=nn.SmoothL1Loss(), device=device)
q_agent.load_weights()
state = torch.tensor(game.print_to_nn(), dtype=torch.int).unsqueeze(0)
state = state.to(device)
action_last_weighted_loss = np.ones(9 * 9) * 10

for i in range(episodes):
    temperature = 10  # 10 * np.exp(-i / 100000)
    if i % 1 == 0:
        action, q_value = q_agent.select_action(state, temperature=temperature)
    else:
        action, q_value = q_agent.select_action(state, policy='weights', weights=action_last_weighted_loss)
    from_column, to_column = int(action // 9), int(action % 9)
    next_state, value, win, termination = game.move(from_column,
                                                    to_column,
                                                    'nn')
    if step % show_step == 0:
        with torch.no_grad():
            q_value_to_print = q_agent.model(state)
    if win:
        print('Win')
        legal_moves = 0
        wins += 1
    elif termination:
        print('End')
        list_legal_moves.append(legal_moves)
        legal_moves = 0
        q_agent.save_weights()
    elif value:
        legal_moves += 1
    value_tensor = torch.tensor(value, dtype=torch.float32)
    foundation_tensor = torch.tensor(to_column == 8, dtype=torch.float32)
    from_foundation_tensor = torch.tensor(from_column == 8, dtype=torch.float32)
    from_waste_to_waste = torch.tensor(from_column == 7, dtype=torch.float32)
    from_waste_to_waste *= torch.tensor(to_column == 7, dtype=torch.float32)
    win_tensor = torch.tensor(win, dtype=torch.float32)
    reward = ((value_tensor * 10 - 5)
              + win_tensor * 1000
              - torch.tensor(termination, dtype=torch.float32) * 0
              - 1
              - from_foundation_tensor * value_tensor * 10
              + foundation_tensor * value_tensor * 10) * (1 - from_waste_to_waste)
    next_state = torch.tensor(next_state, dtype=torch.int).unsqueeze(0).to(device)
    reward = reward.unsqueeze(0).to(device)
    value_tensor = value_tensor.unsqueeze(0).to(device)
    action_tensor = torch.tensor(action.squeeze(1))
    loss = q_agent.calculate_loss(q_value, reward, next_state, value_tensor)
    action_last_weighted_loss[action.squeeze().item()] = (action_last_weighted_loss[action.squeeze().item()] * 0.9 +
                                                          0.1 * loss)

    buffer.append([state, next_state, action_tensor, reward, value_tensor], loss)
    state = next_state
    step += 1
    if step > batch_size and step % train_step == 0:
        temp2 = 1 / 10  # (step % 61) / 29 + 1e-9
        sample = buffer.sample(batch_size, temperature=temp2)
        loss, q_values = q_agent.step(sample)
        losses.append(loss)
    if step % show_step == 0:
        max_q_values = torch.topk(q_value_to_print[0], 5)
        print(f"Steps: {step}, wins: {wins}, "
              # f"Temperature: {temperature:.4f}, "
              f"legal_moves: {legal_moves}, "
              f"mean last 10: {np.mean(list_legal_moves[-10:]):.2f}, "
              f"mean last 50: {np.mean(list_legal_moves[-50:]):.2f}, "
              f"mean last 200: {np.mean(list_legal_moves[-200:]):.2f}, "
              f"loss: {torch.tensor(losses[-1000:]).mean().numpy():.4f}, \n"
              f"action weights: {action_last_weighted_loss[:20].round(2)}, \n"
              f"rewards: {sample[3].cpu().numpy()[:10]},\n"
              f"actions: {sample[2].cpu().numpy()[:10]},\n"
              f"max_q_values: {max_q_values.values.cpu().numpy().round(4)},\n"
              f"index_q_values: {max_q_values.indices.cpu().numpy()}")
