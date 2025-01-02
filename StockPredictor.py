#Program to implement a model for predicting the S and P index using a feedforward neural network
#Note: just a project to learn machine learning. Would not recommend use for actual stock trades.
 
import yfinance as yf
import datetime
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Collects closing data from Yahoo Finance API
ticker_symbol = "SPY"
ticker = yf.Ticker(ticker_symbol)
close_data = ticker.history(period="5y")['Close']
print("Data Collected")


#Processes Data 
def process_data(current_day_index):
    '''
    Converts raw closing price data to a list of the 200 previous close values as a percentage increase from the base day

    Input: 
    index - int - required to be over 200

    Output:
    (list of 200 previous close value percent changes, current value) - tuple
    '''
    if current_day_index < 200:
        raise ValueError('index too low')
    base_day_index = current_day_index - 200
    base_day_value = close_data[base_day_index]
    processed_close_prices = []
    for day in range(base_day_index,current_day_index):
        processed_close_prices.append((close_data[day]/base_day_value) - 1)
    return (processed_close_prices,(close_data[current_day_index])/base_day_value - 1)

#Converts data to a PyTorch tensor
dataset_x = []
dataset_y = []
for i in range(200,1200):
    data = process_data(i)
    dataset_x.append(data[0])
    dataset_y.append(data[1])
dataset_x = torch.tensor(dataset_x, dtype=torch.float32)
dataset_y = torch.tensor(dataset_y, dtype=torch.float32).reshape(-1, 1)
print("Data Processed")

#Initializes neural Network with 3 layers
neural_net = nn.Sequential(
    nn.Linear(200,200),
    nn.ReLU(),
    nn.Linear(200,100),
    nn.ReLU(),
    nn.Linear(100,1)
) 

#Trains the neural network
optimizer = optim.Adam(neural_net.parameters(),lr=0.00005)
for epoch in range(150):
    total_loss = 0
    for i in range(0, 800, 20):
        batch_x = dataset_x[i:i+20]
        batch_y = dataset_y[i:i+20]
        predicted_y = neural_net(batch_x)
        loss_function = nn.MSELoss()
        loss = loss_function(batch_y,predicted_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    print(f'Epoch: {epoch + 1}, Loss: {total_loss}')

#Tests the neural network
total_loss = 0
for i in range(800,1000):
    input = dataset_x[i]
    true_price = dataset_y[i]
    predicted_price = neural_net(input)
    loss_function = nn.MSELoss()
    loss = loss_function(true_price,predicted_price)
    total_loss += loss
print(total_loss)