from word2vec import Word2vec

with open('data.txt', 'r') as f:
    data = f.readlines()

w = Word2vec(data[:50], 8, 2)
w.train(100)
