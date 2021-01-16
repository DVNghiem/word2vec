from tqdm import tqdm
import numpy as np
from vocab import Vocab
from underthesea import word_tokenize
from optimize import Adam
import pickle
import matplotlib.pyplot as plt
import matplotlib


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Word2Vec:

    def __init__(self, data, learning_rate=0.001, embedding_size=128, window_size=2):

        self.embedding_size = embedding_size
        self.window_size = window_size
        self.__createData(data)
        self.w1 = np.random.uniform(-0.8, 0.8,
                                    (self.vocab.vocab_size, self.embedding_size))
        self.w2 = np.random.uniform(-0.8, 0.8,
                                    (self.embedding_size, self.vocab.vocab_size))

        self.learning_rate = learning_rate
        self.optimize_w1 = Adam(learning_rate=self.learning_rate)
        self.optimize_w2 = Adam(learning_rate=self.learning_rate)

    def __createData(self, data):
        self.train_data = []

        self.vocab = Vocab()
        for i in data:
            self.vocab.add(i)

        for i in data:
            sent = self.vocab.processing(i)
            list_char = word_tokenize(sent)
            for index, word in enumerate(list_char):
                target = self.__one_hot_vector(word)
                context = []
                if index == 0:
                    for j in range(1, self.window_size+1):
                        context.append(self.__one_hot_vector(list_char[j]))
                elif index == len(list_char)-1:
                    for j in range(index-self.window_size, len(list_char)-1):
                        context.append(self.__one_hot_vector(list_char[j]))
                else:
                    for j in range(index-1, index-1-self.window_size, -1):
                        if j >= 0:
                            context.append(self.__one_hot_vector(list_char[j]))
                    for j in range(index+1, index+1+self.window_size):
                        if j < len(list_char):
                            context.append(self.__one_hot_vector(list_char[j]))
                self.train_data.append([target, context])

    def __one_hot_vector(self, word):
        vector = np.zeros(self.vocab.vocab_size)
        index_of_word = self.vocab.word2id[word]
        vector[index_of_word] = 1
        return vector

    def __forward_propagation(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y = softmax(u)
        return y, h, u

    def __backward_propagation(self, e, h, x):
        delta_w1 = np.dot(x, np.dot(self.w2, e).T)
        delta_w2 = np.dot(h, e.T)
        self.w1 = self.optimize_w1.optimize(self.w1, delta_w1)
        self.w2 = self.optimize_w2.optimize(self.w2, delta_w2)

    def __loss(self, u, y):
        loss = 0
        for i in y:
            index = np.where(i == 1)[0][0]
            loss += u[index][0]
        loss = -loss
        loss += len(y)*np.log(np.sum(np.exp(u)))
        return loss

    def train(self, epochs=100):
        self.losses = []
        for i in range(1, epochs):
            loss = 0
            for target, context in tqdm(self.train_data):
                target = np.array(target).reshape(-1, 1)
                y_pred, h, u = self.__forward_propagation(target)
                e = np.sum([np.subtract(y_pred, word.reshape(-1, 1))
                            for word in context], axis=0)
                self.__backward_propagation(e, h, target)
                loss += self.__loss(u, context)
            loss /= len(self.train_data)
            self.losses.append(loss)
            print("epoch ", i, " loss = ", loss)

    def word2vec(self, word):
        index = self.vocab.word2id[word]
        vector = self.w1[index]
        return vector

    def get_nearest_word(self, word, number):
        v_w1 = self.word2vec(word)
        word_sim = {}

        for i in range(self.vocab.vocab_size):

            v_w2 = self.w1[i]

            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1)*np.linalg.norm(v_w2)
            theta = theta_sum/theta_den

            word = self.vocab.id2word[i]
            word_sim[word] = theta

        words_sorted = sorted(
            word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:number]:
            print(word, sim)

    def save_model(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    def visualize(self):
        plt.plot(self.losses)
        plt.show()
