from word2vec import Word2Vec
import pickle
with open('data.txt', 'r') as f:
    data = f.readlines()

w = Word2Vec(data[:1], embedding_size=64, window_size=4)
w.train(epochs=100)
# print(w.get_nearest_word('người', 3))
# w.save_model('model.ptkl')
w.visualize()

# with open('model.ptkl', 'rb') as file:
#     w = pickle.load(file)
#     w.get_nearest_word('mẹ', 5)
