import tensorflow as tf
from vocab import Vocab
import numpy as np

tf.compat.v1.disable_eager_execution()


class Word2Vec:

    def __init__(self, data, embedding, window=4):
        self.sess = tf.compat.v1.InteractiveSession()
        self.embedding = embedding
        self.window = window
        self.createData(data)
        self.initialParameter()

    def initialParameter(self):

        weight_initer = tf.compat.v1.truncated_normal_initializer(
            mean=0.0, stddev=0.01)
        self.w1 = tf.compat.v1.get_variable(name="W1", dtype=tf.float32, shape=[
            self.vocab.vocab_size, self.embedding], initializer=weight_initer)
        self.w2 = tf.compat.v1.get_variable(name="W2", dtype=tf.float32, shape=[
            self.embedding, self.vocab.vocab_size], initializer=weight_initer)

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
                                for x in range(i+1, self.window+1)]
            elif i == len(self.vocab.corpus)-1:
                context_word = [self.vocab.corpus[x] for x in range(
                    self.vocab.len_corpus-2, self.vocab.len_corpus-2-self.window, -1)]
            else:
                before = index_target_word - 1
                for x in range(before, before-self.window, -1):
                    if x >= 0:
                        context_word.extend([self.vocab.corpus[x]])
                after = index_target_word+1
                for x in range(after, after+self.window):
                    if x < len(self.vocab.corpus):
                        context_word.extend([self.vocab.corpus[x]])

            target_vector, context_vector = self.one_hot_vector(
                target_word, context_word)
            self.train_x.append(target_vector)
            self.train_y.append(context_vector)
        self.train_x = np.array(self.train_x, dtype=np.float32)
        self.train_y = np.array(self.train_y, dtype=np.float32)

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
        h = tf.matmul(x, self.w1)
        u = tf.matmul(h, self.w2)
        return tf.nn.softmax(u), h, u

    def backward_propagation(self, e, h, x):
        delta_w1 = tf.matmul(x.T, tf.matmul(e, self.w1))
        delta_w2 = tf.matmul(tf.transpose(h), e)
        self.w1 = self.w1 - self.learning_rate*delta_w1
        self.w2 = self.w2 - self.learning_rate*delta_w2

    def loss(self, u, y):
        loss = 0
        c = 0
        for i in np.where(y == 1)[0]:
            loss += u[i][0]
            c += 1
        loss = -loss+c*tf.math.log(tf.reduce_sum(tf.math.exp(u)))
        return loss

    def train(self, epochs=10, batch_size=128, learning_rate=0.001):
        self.learning_rate = learning_rate
        nb_batch = len(self.train_x)//batch_size
        for i in range(epochs):
            self.losses = 0
            for j in range(nb_batch):
                batch_x = self.train_x[j *
                                       batch_size:(j+1)*batch_size]

                batch_y = self.train_y[j *
                                       batch_size:(j+1)*batch_size]
                y_pred, h, u = self.forward_propagation(batch_x)
                e = y_pred - batch_y
                self.backward_propagation(e, h, batch_x)
                self.losses += self.loss(u, batch_y)
            self.losses /= nb_batch
            print(f'epoch: {i+1}, loss: {self.losses}')
