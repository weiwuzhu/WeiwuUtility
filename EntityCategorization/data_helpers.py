import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),;!?&\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"&", " & ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(training_data_file, class_index_file, isEval=False, hasHeader=True):
    inputFile = open(training_data_file, 'r', encoding='utf-8')

    schema = {}
    isFirst = True
    examples = []
    labels = []
    for line in inputFile:
        fields = line[:-1].split('\t')
        if isFirst and hasHeader:
            for i in range(len(fields)):
                schema[fields[i]] = i
            isFirst = False
            continue

        examples.append(fields[schema['YelpCategory']])
        labels.append(fields[schema['ClickedCategory']])
    
    # Split by words
    x = [clean_str(sent) for sent in examples]

    # Generate labels
    categories = set()
    for l in labels:
        for c in l.split(';'):
            categories.add(c)

    categories = sorted(list(categories))
    classes = {}
    if not isEval:
        classIndexFile = open(class_index_file, 'w', encoding='utf-8')
        index = 0
        for c in categories:
            classes[c] = index
            classIndexFile.write(c + '\t' + str(index) + '\n')
            index += 1
    else:
        classIndexFile = open(class_index_file, 'r', encoding='utf-8')
        for line in classIndexFile:
            fields = line[:-1].split('\t')
            classes[fields[0]] = int(fields[1])

    y = []
    for l in labels:
        label = [0] * len(classes)
        for c in l.split(';'):
            label[classes[c]] = 1
        y.append(label)

    return [x, np.array(y)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def label2string(labels, class_index_file):
    class_index = {}
    for line in list(open(class_index_file, "r").readlines()):
        fields = line[:-1].split('\t')
        class_index[int(fields[1])] = fields[0]
    
    result = []
    for label in labels:
        predictClass = []
        index = 0
        for i in label:
            if i == 1:
                predictClass.append(class_index[index])
            index += 1
        result.append(';'.join(predictClass))
    return result

if __name__ == '__main__':
    [x, y] = load_data_and_labels("D:\Code\WeiwuUtility\EntityCategorization\data\YelpClick\YelpClickTrainingData.tsv", "D:\Code\WeiwuUtility\EntityCategorization\data\YelpClick\YelpClickCategoryIndex.tsv")
    print(y[0])
