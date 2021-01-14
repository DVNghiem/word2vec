import numpy as np
from vocab import Vocab


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Word2vec:

    def __init__(self, data, embedding_size, window_size):

        self.embedding_size = embedding_size
        self.window_size = window_size
        self.createData(data)
        self.w1 = np.random.uniform(-0.8, 0.8,
                                    (self.vocab.vocab_size, self.embedding_size))
        self.w2 = np.random.uniform(-0.8, 0.8,
                                    (self.embedding_size, self.vocab.vocab_size))

        self.learning_rate = 0.001

    def createData(self, data):
        self.train_x = []
        self.train_y = []

        self.vocab = Vocab()
        for i in data:
            self.vocab.add(i)

        for i, word in enumerate(self.vocab.corpus):
            index_target_word = i
            target_word = word
            context_word = []

            if i == 0:
                context_word = [self.vocab.corpus[x]
                                for x in range(i+1, self.window_size+1)]
            elif i == len(self.vocab.corpus)-1:
                context_word = [self.vocab.corpus[x] for x in range(
                    self.vocab.len_corpus-2, self.vocab.len_corpus-2-self.window_size, -1)]
            else:
                before = index_target_word - 1
                for x in range(before, before-self.window_size, -1):
                    if x >= 0:
                        context_word.extend([self.vocab.corpus[x]])
                after = index_target_word+1
                for x in range(after, after+self.window_size):
                    if x < len(self.vocab.corpus):
                        context_word.extend([self.vocab.corpus[x]])

            target_vector, context_vector = self.one_hot_vector(
                target_word, context_word)
            self.train_x.append(target_vector)
            self.train_y.append(context_vector)
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)

    def one_hot_vector(self, target_word, context_word):
        target = np.zeros(self.vocab.vocab_size)
        index_of_word = self.vocab.word2id[target_word]
        target[index_of_word] = 1
        context = np.zeros(self.vocab.vocab_size)
        for word in context_word:
            index = self.vocab.word2id[word]
            context[index] = 1
        return target, context

    def forward_propagation(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y = softmax(u)
        return y, h, u

    def backward_propagation(self, e, h, x):
        delta_w1 = np.dot(x, np.dot(self.w2, e).T)
        delta_w2 = np.dot(h, e.T)
        self.w1 = self.w1 - self.learning_rate*delta_w1
        self.w2 = self.w2 - self.learning_rate*delta_w2

    def loss(self, u, y):
        loss = 0
        c = 0
        for i in np.where(y == 1)[0]:
            loss += u[i][0]
            c += 1
        loss = -loss
        u = np.array(u, dtype=np.float128)
        loss += c*np.log(np.sum(np.exp(u)))
        return loss

    def train(self, epochs=100, batch_size=128):
        nb_batch = len(self.train_x)//batch_size
        for i in range(1, epochs):
            loss = 0
            for j in range(nb_batch):
                batch_x = self.train_x[j *
                                       batch_size:(j+1)*batch_size].T

                batch_y = self.train_y[j *
                                       batch_size:(j+1)*batch_size].T
                y_pred, h, u = self.forward_propagation(batch_x)
                e = y_pred-batch_y

                self.backward_propagation(e, h, batch_x)
                loss += self.loss(u, batch_y)
            loss /= nb_batch
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
