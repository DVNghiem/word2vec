from word2vec_tf import Word2Vec

with open('data.txt', 'r') as f:
    data = f.readlines()

w = Word2Vec(data[:10], 8, 2)
w.train(epochs=2)
# print(w.get_nearest_word('yêu', 5))
