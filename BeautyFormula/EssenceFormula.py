import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

material_index_file = 'material_index.txt'
effect_index_file = 'effect_index.txt'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 10240)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(10240, 10240)
        self.fc3 = nn.Linear(10240, 30)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        y = self.softmax(x)
        return y

def normStr(s):
    return s.strip().strip('"').strip('\n')

def processData(path):
    data = []
    file = open(path, 'r', encoding='utf8')
    for line in file:
        if not line.startswith('#'):
            items = line.strip('\n').split('\t')
            data.append(items)
    file.close()

    result = []
    materials, effects = {}, {}
    len_materials, len_effects = 30, 8
    for i in range(len(data)):
        length = len(data[i])
        if i < len_materials:
            materials[data[i][0]] = i
            for j in range(1, length):
                if len(result) < j:
                    result.append([[0]*len_materials, [0]*len_effects])
                result[j-1][0][i] = 0 if not data[i][j] else float(data[i][j].strip('%'))/100
        elif len_materials <= i < len_materials + len_effects:
            new_i = i - len_materials
            effects[data[i][0]] = new_i
            for j in range(1, length):
                result[j-1][1][new_i] = 0 if not data[i][j] else float(data[i][j])

    file = open(material_index_file, 'w', encoding='utf8')
    for m in materials:
        file.write(m + '\t' + str(materials[m]) + '\n')
    file.close()
    file = open(effect_index_file, 'w', encoding='utf8')
    for e in effects:
        file.write(e + '\t' + str(effects[e]) + '\n')
    file.close()
    
    return result

def orgnize_data(data, train_sample_ration=0.8):
    result = []
    for d in data:
        x = d[1]
        y = d[0]
        result.append([x, y])
    random.shuffle(result)

    train_data_index = int(train_sample_ration * len(result))
    split_train_data, split_test_data = result[:train_data_index], result[train_data_index:]
    train_data, test_data = data2tensor(split_train_data), data2tensor(split_test_data)
    return train_data, test_data

def data2tensor(data, batch_size=33):
    result = []
    length = len(data)
    i = 0
    while i < length:
        batch = data[i:min(i+batch_size, length)]
        i += batch_size
        x = torch.tensor([j[0] for j in batch])
        y = torch.tensor([j[1] for j in batch])
        result.append([x, y])
    return result

def my_loss(output, target, data):
    loss = torch.mean(((output - target)*(data[:,5:]))**2)
    return loss

def train(train_data, model, optimizer, criterion, num_epoch):
    model.train()
    losses = []
    for e in range(num_epoch):
        running_loss = 0
        for i, data in enumerate(train_data):
            x, y = data
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            #loss = my_loss(output, y, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        losses.append(running_loss/len(train_data))

        if e % 10 == 0:
            now = datetime.now()
            dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
            print('%s [epoch %d] loss: %.5f' % (dt_string, e, losses[-1]))
    return losses

def eval(test_data, model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

    result = {}
    with torch.no_grad():
        output = model(test_data)
        output = output.numpy()[0]

    result = beautify_result(output)
    return result

def eval_single(model, path, data, id):
    model.load_state_dict(torch.load(path))
    model.eval()

    running_loss = 0
    test_data = data2tensor([[data[id][1], data[id][0]]])
    with torch.no_grad():
        for i, new_data in enumerate(test_data):
            x, y = new_data
            output = model(x)
            loss = criterion(output, y)
            #loss = my_loss(output, y, x)
            running_loss = loss.item()
            output = output.numpy()[0]
    #arr = [(k, result[k]) for k in result]
    #value = torch.from_numpy(np.array([i[1] for i in arr]))
    #sm = nn.Softmax()
    #norm_value = sm(value).numpy()
    #result = {arr[i][0]:norm_value[i] for i in range(len(arr))}
    #sum_p = sum([result[k] for k in result])
    #ratio = 1.0 / sum_p
    #for k in result:
    #    result[k] *= ratio

    print('id,', id, 'loss:', running_loss)
    print([[effects[i], data[id][1][i]] for i in range(len(effects))])
    result = beautify_result(output)
    return result

def beautify_result(formula, round_num=2, effect=None):
    materials, effects = {}, {}
    file = open(material_index_file, 'r', encoding='utf8')
    for line in file:
        if not line.startswith('#'):
            items = line.split('\t')
            if len(items) == 2:
                name = items[0]
                index = items[1]
                materials[int(index)] = name
    file.close()
    file = open(effect_index_file, 'r', encoding='utf8')
    for line in file:
        if not line.startswith('#'):
            items = line.split('\t')
            if len(items) == 2:
                name = items[0]
                index = items[1]
                effects[int(index)] = name
    file.close()

    result = {}
    if effect:
        for i in range(len(effect)):
            result[effects[i]] = effect[i]
        print('\neffect==========')
    else:
        for i in range(len(formula)):
            result[materials[i]] = round(formula[i]*100, round_num)
        print('\nfomular==========')
    print(sorted(result.items(), key=lambda x: x[1], reverse=True))
    return result

def run(effect):
    print('loading data...')
    data = processData('guihua.txt')
    train_data, test_data = orgnize_data(data, 1.0)

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    #criterion = nn.KLDivLoss()
    criterion = nn.MSELoss()
    model_path = 'nn_model.pth'
    """
    print('training started...')
    losses = train(train_data, net, optimizer, criterion, 10000)
    print('training finished...')
    plt.plot(losses)
    torch.save(net.state_dict(), model_path)
    print('model saved in', model_path)
    """
    for i in range(len(data)):
        if data[i][1] == effect:
            return beautify_result(data[i][0])
            # return eval_single(net, model_path, data, i)
        
    return eval(torch.tensor([[float(num) for num in effect]]), net, model_path)
    print('evaluation finished...')

if __name__ == "__main__":
    #effect = [2,2,2,2,2,2,2,0]
    effect = [5,5,5,5,5,5,5,0]
    result = run(effect)
    print('done...')
