from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


#textrank algo

def read_paragraph(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    para = filedata[0].split(". ")
    sentences = []

    for sentence in para:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(s1,s2,stopwords=None):
    if stopwords is None:
        stopwords = []
 
    s1 = [a.lower() for a in s1]
    s2 = [a.lower() for a in s2]
 
    all_words = list(set(s1 + s2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
 
    for w in s1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    for w in s2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: 
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name):
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences =  read_paragraph(file_name)

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
  
    summarize_text.append(" ".join(ranked_sentence[0][1]))

    print("Summarized Text: \n", ". ".join(summarize_text))

print("Enter file name: ")
file_name = input()
generate_summary( file_name)
