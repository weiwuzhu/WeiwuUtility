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
        self.fc1 = nn.Linear(320, 10240)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(10240, 10240)
        self.fc3 = nn.Linear(10240, 315)
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
    data = {}
    count = 0
    file = open(path, 'r', encoding='utf8')
    baddata = []
    for line in file:
        if not line.startswith('#'):
            items = line.split('\t')
            if len(items) == 4:
                id = items[0]
                data[id] = []
                data[id].append(normStr(items[1]))
                data[id].append(normStr(items[2]))
                data[id].append(normStr(items[3]))
                count += 1
            else:
                baddata.append(line)

    #print(baddata)
    file.close()
    print('total valid data count', count)
    index, materials = 0, {}
    index1, effects = 0, {}
    for id in data:
        materialstr = data[id][1]
        for item in materialstr.split(','):
            if item and item not in materials:
                materials[item] = index
                index += 1
        effectstr = data[id][0]
        for item in effectstr.split(','):
            if item:
                key = item.split(':')[0]
                if key not in effects:
                    effects[key] = index1
                    index1 += 1
    file = open(material_index_file, 'w', encoding='utf8')
    for m in materials:
        file.write(m + '\t' + str(materials[m]) + '\n')
    file.close()
    file = open(effect_index_file, 'w', encoding='utf8')
    for e in effects:
        file.write(e + '\t' + str(effects[e]) + '\n')
    file.close()
    result = {}
    for id in data:
        if id == 'HD1901011':
            continue
        result[id] = []
        effect = data[id][0]
        ev = [0] * len(effects)
        for e in effect.split(','):
            if e:
                key, value = e.split(':')[0], e.split(':')[1]
                ev[effects[key]] = float(value)
        result[id].append(ev)
        fomular = data[id][2]
        fv = [0] * len(materials)
        for f in fomular.split(','):
            if f and ':' in f:
                key, value = f.split(':')[0], float(f.split(':')[1])/100
                if value >= 0.005:
                    fv[materials[key]] = value
        result[id].append(fv)
        material = data[id][1]
        mv = [0] * len(materials)
        for m in material.split(','):
            if m and fv[materials[m]] > 0:
                mv[materials[m]] = 1
        result[id].append(mv)
    return result

def orgnize_data(data, train_sample_ration=0.8):
    result = []
    for id in data:
        x = data[id][0] + data[id][1]
        y = data[id][2]
        result.append([x, y])
    random.shuffle(result)

    train_data_index = int(train_sample_ration * len(result))
    split_train_data, split_test_data = result[:train_data_index], result[train_data_index:]
    train_data, test_data = data2tensor(split_train_data), data2tensor(split_test_data)
    return train_data, test_data

def data2tensor(data, batch_size=10):
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
            #loss = criterion(output, y)
            loss = my_loss(output, y, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        losses.append(running_loss/len(train_data))

        if e % 100 == 0:
            now = datetime.now()
            dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
            print('%s [epoch %d] loss: %.5f' % (dt_string, e, losses[-1]))
    return losses

def eval(test_data, model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    correct, total = 0, 0
    result = []
    with torch.no_grad():
        for i, data in enumerate(test_data):
            x, y = data
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.data.numpy())
            print('loss: %.3f' % (losses[-1]))
    return losses

def eval_single(model, path, data, id):
    model.load_state_dict(torch.load(path))
    model.eval()

    materials = {}
    file = open(material_index_file, 'r', encoding='utf8')
    for line in file:
        if not line.startswith('#'):
            items = line.split('\t')
            if len(items) == 2:
                name = items[0]
                index = items[1]
                materials[int(index)] = name
    file.close()

    result = {}
    running_loss = 0
    x, y, m = data[id][0] + data[id][1], data[id][2], data[id][1]
    test_data = data2tensor([[x, y]])
    with torch.no_grad():
        for i, data in enumerate(test_data):
            x, y = data
            output = model(x)
            #loss = criterion(output, y)
            loss = my_loss(output, y, x)
            running_loss = loss.item()
            output = output.numpy()[0]
            for i in range(len(output)):
                if m[i] == 1:
                    result[materials[i]] = output[i]
    #arr = [(k, result[k]) for k in result]
    #value = torch.from_numpy(np.array([i[1] for i in arr]))
    #sm = nn.Softmax()
    #norm_value = sm(value).numpy()
    #result = {arr[i][0]:norm_value[i] for i in range(len(arr))}
    sum_p = sum([result[k] for k in result])
    ratio = 1.0 / sum_p
    for k in result:
        result[k] *= ratio

    print('id,', id, 'loss:', running_loss)
    print(sorted(result.items(), key=lambda x: x[1], reverse=True))

print('loading data...')
data = processData('trainingdata-20200518.txt')
train_data, test_data = orgnize_data(data, 1.0)

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
#criterion = nn.KLDivLoss()
criterion = nn.MSELoss()
model_path = 'nn_model.pth'

print('started training...')
losses = train(train_data, net, optimizer, criterion, 100)
print('finished training...')
plt.plot(losses)
torch.save(net.state_dict(), model_path)
print('model saved in', model_path)
""""""
eval_single(net, model_path, data, 'HD1901010')
