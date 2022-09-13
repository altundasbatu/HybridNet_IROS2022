import torch
import torch.nn as nn

class LSTM_CellLayer(nn.Module):
    def __init__(self, input_shape, hidden_shape):
        super(LSTM_CellLayer, self).__init__()
        self.n_input = input_shape
        self.n_hidden = hidden_shape
        self.cell = nn.LSTMCell(input_shape, hidden_shape)
        
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(batch_size, self.n_hidden, requires_grad=True)
        cell_state = torch.zeros(batch_size, self.n_hidden, requires_grad=True)
        return (hidden_state, cell_state)
        
    def forward(self, x, hidden):
        batch_size = x.shape[0]
        hx, cx = hidden
        # input = x.flatten(start_dim=1) # flatten everything but the batch index
        hx, cx = self.cell(x, (hx, cx))
        return hx, (hx, cx)

class LSTM_CellLayer2(nn.Module):
    def __init__(self, input_shape, hidden_shape):
        super(LSTM_CellLayer2, self).__init__()
        # self.decoder = nn.Sequential(
        #     nn.Conv1d(1, 32, 33, stride=1),
        #     nn.ReLU(),
        #     # nn.Conv1d(32, 32, 33, stride=1),
        #     # nn.ReLU(),
        #     # nn.Linear(input_shape, input_shape * 32),
        #     # nn.ReLU(),
        #     # nn.Linear(input_shape * 32, input_shape * 32),
        #     # nn.ReLU(),
        #     # nn.Linear(input_shape * 32, input_shape * 32),
        #     # nn.ReLU()
        # )
        
        self.n_input = input_shape * 16
        self.n_hidden = hidden_shape
        self.cell = nn.LSTMCell(self.n_input, self.n_hidden)
        
        # self.encoder = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(input_shape * 4, input_shape * 2),
        #     nn.ReLU(),
        #     nn.Linear(input_shape * 2, input_shape),
        #     nn.ReLU()
        # )
        
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(batch_size, self.n_hidden, requires_grad=True)
        cell_state = torch.zeros(batch_size, self.n_hidden, requires_grad=True)
        return (hidden_state, cell_state)
    
    def forward(self, x, hidden):
        batch_size = x.shape[0]
        hx, cx = hidden
        # print(batch_size)
        x1 = self.decoder(torch.unsqueeze(x, 1))
        x1 = torch.flatten(x1, 1)
        # print(x1.shape)
        # print(x1.shape, hidden[0].shape, hidden[1].shape)
        hx, cx = self.cell(x1, (hx, cx))
        # x2 = self.encoder(hx)
        return hx, (hx, cx)


if __name__ == '__main__':
    # n_features = 2 # this is number of parallel inputs
    # n_timesteps = 1 # this is number of timesteps

    # # convert dataset into input/output
    # X, y = split_sequences(dataset, n_timesteps)
    # print(X.shape, y.shape)

    # # create NN
    # lstm_net = LSTM_Layer(n_features,n_timesteps)
    # criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
    # optimizer = torch.optim.Adam(lstm_net.parameters(), lr=1e-1)

    # train_episodes = 500
    # batch_size = 16
    rnn = LSTM_CellLayer((2,5), 20) # (input_size, hidden_size)
    input = torch.randn(2, 3, 2, 5) # (time_steps, batch, input_size[0], input_size[1])
    hx, cx = rnn.init_hidden(3)
    output = []
    for i in range(input.size()[0]):
        hx, cx = rnn(input[i], (hx, cx))
        output.append(hx)
    output = torch.stack(output, dim=0)
    print(output.shape)
    pass