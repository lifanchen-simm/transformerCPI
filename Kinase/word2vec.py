# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/4/30 16:10
@author: LiFan Chen
@Filename: word2vec.py
@Software: PyCharm
"""
from gensim.models import Word2Vec
import pandas as pd
import numpy as np


def seq_to_kmers(seq, k=3):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


class Corpus(object):
    """ An iteratable for training seq2vec models. """

    def __init__(self, dir, ngram):
        self.df = pd.read_csv(dir)
        self.ngram = ngram

    def __iter__(self):
        for sentence in self.df.Seq.values:
            yield seq_to_kmers(sentence, self.ngram)


def get_protein_embedding(model,protein):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((len(protein), 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
    return vec


if __name__ == "__main__":

    sent_corpus = Corpus("dataset/celegans_uniprot.csv",3)
    model = Word2Vec(size=100, window=5, min_count=1, workers=6)
    model.build_vocab(sent_corpus)
    model.train(sent_corpus,epochs=30,total_examples=model.corpus_count)
    model.save("word2vec_30_celegans.model")

    """
    model = Word2Vec.load("word2vec_30.model")
    vector = get_protein_embedding(model,seq_to_kmers("MSPLNQSAEGLPQEASNRSLNATETSEAWDPRTLQALKISLAVVLSVITLATVLSNAFVLTTILLTRKLHTPANYLIGSLATTDLLVSILVMPISIAYTITHTWNFGQILCDIWLSSDITCCTASILHLCVIALDRYWAITDALEYSKRRTAGHAATMIAIVWAISICISIPPLFWRQAKAQEEMSDCLVNTSQISYTIYSTCGAFYIPSVLLIILYGRIYRAARNRILNPPSLYGKRFTTAHLITGSAGSSLCSLNSSLHEGHSHSAGSPLFFNHVKIKLADSALERKRISAARERKATKILGIILGAFIICWLPFFVVSLVLPICRDSCWIHPALFDFFTWLGYLNSLINPIIYTVFNEEFRQAFQKIVPFRKAS"))
    print(vector.shape)
    """