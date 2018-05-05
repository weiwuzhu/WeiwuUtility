import numpy as np
import re
import itertools
import gensim
from collections import Counter
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
#from scipy import spatial

UNK_TOKEN = '<UNK>' # unknown word

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

def convert_word_to_id(data_list):
    words = {}
    for line in data_list:
        tokens = set(line.split())
        for token in tokens:
            if token in words:
                words[token] += 1
            else:
                words[token] = 1

    vocab_list = sorted(words, key=words.get, reverse=True)
    word2id = {}
    index = 1   # Reserve the first item for unknown words
    for v in vocab_list:
        word2id[v] = index
        index += 1
    return word2id

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

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            print("word length: " + str(len(word)))
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs, layer1_size

def load_bin_vec_gensim(fname, vocab):
    word_vecs = {}
    firstWord = True
    emb_size = 0
    model = KeyedVectors.load_word2vec_format(fname, binary=True)
    #print(model.similarity('restaurant', 'restaurants'))
    #print(model.similarity('restaurant', 'hotel'))
    #print(model.similarity('restaurant', 'bar'))
    for word in vocab:
        if word in model:
            word_vecs[word] = model[word]
            if firstWord:
                emb_size = len(word_vecs[word])
                firstWord = False
    return word_vecs, emb_size

def prepare_pretrained_embedding(fname, word2id):
    print('Reading pretrained word vectors from file ...')
    word_vecs, emb_size = load_bin_vec_gensim(fname, word2id)
    word_vecs = add_unknown_words(word_vecs, word2id, emb_size)
    embedding = np.zeros(shape=(len(word2id)+1, emb_size), dtype='float32')
    for w in word2id:
        embedding[word2id[w]] = word_vecs[w]
    
    # Add unknown word into the vocabulary
    word2id[UNK_TOKEN] = 0
    print('Generated embeddings with shape ' + str(embedding.shape))
    return embedding, word2id

def add_unknown_words(word_vecs, vocab, emb_size=300):
    """
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, emb_size)
    return word_vecs

if __name__ == '__main__':
    [x, y] = load_data_and_labels("data\YelpClick\YelpClickTrainingData.tsv", "data\YelpClick\YelpClickCategoryIndex.tsv")
    word2id = convert_word_to_id(x)
    embedding, word2id = prepare_pretrained_embedding("data\WordVector\GoogleNews-vectors-negative300.bin", word2id)
    np.save("data/WordVector/vocab.npy", word2id)
    np.save('data/WordVector/emb.npy', embedding)
    print("dataset created!")
