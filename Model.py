import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module) :
    
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super().__init__()
        # Первый скрытый слой
        self.linear1 = nn.Linear(input_size, hidden_size1)
        # Второй скрытый слой
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        # Третий скрытый слой
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        # Выходной слой
        self.linear4 = nn.Linear(hidden_size3, output_size)
    
    def forward(self, x):
        # Прямой проход через все слои с активацией ReLU
        x = F.relu(self.linear1(x))  # 1-й слой (input -> hidden1)
        x = F.relu(self.linear2(x))  # 2-й слой (hidden1 -> hidden2)
        x = F.relu(self.linear3(x))  # 3-й слой (hidden2 -> hidden3)
        x = self.linear4(x) # выходной слой (hidden3 -> output)
        return x
    
    def save(self, file_name = 'model.pth') :
        model_folder_path = './model'
        if not os.path.exists :
            os.makedirs(model_folder_path)

            file_name = os.path.join(model_folder_path, file_name)
            torch.save(self.state_dict(), file_name)


class Qtrainer :
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_steps(self, states, actions, reward, next_states) :
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

       
        pred = self.model(states)

        target = pred.clone()
        for idx in range(0, len(reward)):
            # Используем только награду из reward[idx]
            Q_new = reward[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            # Обновляем целевое значение в соответствии с действиями
            target[idx][torch.argmax(actions[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()