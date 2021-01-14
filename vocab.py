import underthesea
import os
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# STOP_WORD = os.path.join(BASE_DIR, 'word2vec/vietnamese-stopwords-dash.txt')


class Vocab:

    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        self.corpus = []

    def add(self, sentence):
        sentence = self.processing(sentence)
        list_char = underthesea.word_tokenize(sentence)
        for i in list_char:
            self.corpus.append(i)
            if i not in self.word2id.values():

                self.word2id[i] = self.vocab_size
                self.id2word[self.vocab_size] = i
                self.vocab_size += 1
        self.len_corpus = len(self.corpus)

    def processing(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.replace('.', ' ')
        sentence = sentence.replace(',', ' ')
        sentence = sentence.replace('?', ' ')
        sentence = sentence.replace('!', ' ')
        sentence = sentence.replace('(', ' ')
        sentence = sentence.replace(')', ' ')
        sentence = sentence.replace('\\', ' ')
        sentence = sentence.replace('//', ' ')
        sentence = sentence.replace('+', ' ')
        sentence = sentence.replace('-', ' ')
        sentence = sentence.replace('%', ' ')
        sentence = sentence.replace('$', ' ')
        sentence = sentence.replace('@', ' ')
        sentence = sentence.replace(':', ' ')
        sentence = sentence.replace(';', ' ')
        sentence = sentence.replace('"', ' ')

        return sentence
