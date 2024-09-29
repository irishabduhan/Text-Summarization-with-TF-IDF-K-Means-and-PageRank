import nltk
from nltk.tokenize import word_tokenize
import networkx as nx
import string
import math
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download("punkt")

#importing and opening then reading file
file_path = 'input.txt'
with open(file_path, 'r') as file:
    initial_content = file.read()
    #print(content)
    initial_content = initial_content.split('\n')
    #print(initial_content)

#initial_content -> content
#removing empty sentences
content = []
for line in initial_content:
    if line.strip() != '':
        content.append(line)

#content

final_content = []
#make translation table for making pucuation symbol to NULL
for line in initial_content:
    translator = str.maketrans('', '', string.punctuation)
    new_line = line.translate(translator)
    final_content.append(new_line)

#final_content

#converting words to lower case
#then tokenizing the lower case sentences then storing in content
content = []
for line in final_content:
    new_line = line.lower()
    tokenized_line = word_tokenize(new_line)
    content.append(tokenized_line)

#content

stop_words = set(stopwords.words('english'))
#content -> final_content
#removing stopwords 
final_content = []
for line in content:
    new_line = []
    for word in line:
        if word not in stop_words:            
            new_line.append(word)
            
    final_content.append(new_line)

#final_content

#converting words to their base form
lemmatizer = WordNetLemmatizer()
processed_list = []

for line in final_content:
    new_line = []
    for word in line:
        pos = wordnet.synsets(word)[0].pos() if wordnet.synsets(word) else 'n'
        dict_word = lemmatizer.lemmatize(word, pos=pos)
        new_line.append(dict_word)
        
    processed_list.append(new_line)

new_processed_list = []
for line in processed_list:
    new_list = " ".join(line)
    new_processed_list.append(new_list)

# new_processed_list




#T2 starts

sentence_list = processed_list
#sentence_list

# finding dimensions of tf-idf matrix dimension

word_list = []
for list in sentence_list:
    for word in list:
        word_list.append(word)

unique_word_list = set(word_list)

no_of_column = len(sentence_list)
print("no of sentences: ",no_of_column)
no_of_rows = len(unique_word_list)

#initializing tf-idf matrix
matrix = []
for row_word in unique_word_list:
    rowi = []
    
    #findig the no of sentences in which row_word occurs
    word_in_sentences = 0
    for list in sentence_list:
        for word in list:
            if word == row_word:
                word_in_sentences += 1
                break
                
    for list in sentence_list:
        #finding no of repetition of word in sentence
        counting_list = Counter(list)
        tf_value = counting_list[row_word]
        
        if (tf_value > 0) and (word_in_sentences) > 0:
            idf_value = math.log((no_of_column / word_in_sentences), 2)
        else:
            idf_value = 0

        tf_idf_value = tf_value * idf_value
        rowi.append(tf_idf_value)

    matrix.append(rowi) 

# print(unique_word_list)
# for row in matrix:
#     print(row)





#   TASK 3



#creating graph and adding nodes to it
graph = nx.Graph()

for i in range(no_of_column):
    graph.add_node(i)

# matrix = matrix.T
def transpose_matrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    
    return transposed

transposed_matrix = transpose_matrix(matrix)
# code for printing transposed matrix

# for row in transposed_matrix:
#     print(row)

def cosine_similarity(transposed_matrix, sentence1_idx, sentence2_idx):
    dot_product = 0
    magnitude1 = 0
    magnitude2 = 0

    for tfidf1, tfidf2 in zip(transposed_matrix[sentence1_idx], transposed_matrix[sentence2_idx]):
        dot_product += tfidf1 * tfidf2
        magnitude1 += tfidf1 ** 2
        magnitude2 += tfidf2 ** 2

    magnitude1 = math.sqrt(magnitude1)
    magnitude2 = math.sqrt(magnitude2)

    if magnitude1 > 0 and magnitude2 > 0:
        similarity = dot_product / (magnitude1 * magnitude2)
    else:
        similarity = 0
    
    return similarity


# Create edges between vertices
for u in range(no_of_column):
    # only make edge to nest ones to Avoid creating self-loops and duplicate edges
    for v in range(u + 1, no_of_column):  
        similarity = cosine_similarity(transposed_matrix, u, v)
        graph.add_edge(u, v, weight = similarity)


         

def calculate_page_rank(graph):
    page_rank_values = nx.pagerank(graph)
    return page_rank_values

page_rank_values = calculate_page_rank(graph)

# Take user input for the number of top nodes
n = int(input("Enter the number of top nodes to display: "))
#n = no_of_column
if n> no_of_column:
    print("Error: please pass value of n <= ", no_of_column - 1)
    exit()
else:
    sorted_nodes = sorted(page_rank_values, key=page_rank_values.get, reverse=True)
    for ind in range(n):
        print("Node ", ind, " : ", sentence_list[ind])



#performing MMR function:

non_selected = {}
selected = {}
for node in sorted_nodes:
    page_rank = page_rank_values[node]
    key = node
    non_selected[key] = page_rank 
    
for key, value in non_selected.items():
    selected[key] = value
    del non_selected[key]
    break


#letting lemda
print("number of top nodes value must be less than: ", no_of_column)
n = int(input("Enter the number of top nodes to choose for MMR: "))
if n <= no_of_column:
    lemda = float(input("Enter the value of lembda: "))
    
    
    for i in range(n-1):
        #updating pagerank of non_selected
        for key, value in non_selected.items():
            max_similarity = 0
            for subkey, subvalue in selected.items():
                similarity = cosine_similarity(transposed_matrix, key, subkey)
                if similarity > max_similarity:
                    max_similarity = similarity
        
            non_selected[key] = lemda * (value) - (1-lemda) * max_similarity
        
        key_tobe_pop = 0
        max_value = 0
        for key, value in non_selected.items():
            if max_value < value:
                max_value = value
                key_tobe_pop = key
        
        for key, value in non_selected.items():
            if max_value == value and key_tobe_pop == key:
                    selected[key] = value
                    del non_selected[key]
                    break
        
    
    for key, value in selected.items():
        print("Node ", key, " : ", value)
else:
    print("Error: the number of top nodes exceeds ", no_of_column)
    exit()



#       ***********K mean clustering**********




def calc_mean(cluster):
    sum = np.zeros(no_of_rows)
    for x in cluster:
        sum += np.array(transposed_matrix[x])

    if len(cluster)>0:
        centroid_vector = sum/len(cluster)
    else:
        centroid_vector = sum
    return (centroid_vector.tolist())

# Calculate cosine similarity between two vectors
def cosine_similarity_k(index, vector2):
    vector1 = np.array(transposed_matrix[index])
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    if magnitude1 > 0 and magnitude2 > 0:
        similarity = dot_product / (magnitude1 * magnitude2)
    else:
        similarity = 0
    
    return similarity


#sentence_list
k = int(input("Enter the number of clusters to form: "))
centroid_cluster_list_index = random.sample(sorted_nodes, k)
centroid_cluster_list = []
for index in centroid_cluster_list_index:
    new_list = []
    for num in transposed_matrix[index]:
        new_list.append(num)
    centroid_cluster_list.append(new_list)

# for list in centroid_cluster_list:
#     print(list)


flag = 0
iteration = 0
while flag == 0 :
    cluster_list = []
    for node in centroid_cluster_list:
        cluster = []
        #cluster.append(node)
        cluster_list.append(cluster)
    
    for node in sorted_nodes:
        max = -2
        belong_to = -1
        index = -1
            
        for c_node in centroid_cluster_list:
            index += 1
            similar = cosine_similarity_k(node, np.array(c_node))
            if max < similar:
                max = similar
                belong_to = index
        cluster_list[belong_to].append(node)
        # print(node, " added in ", belong_to, " cluster ")
    
    
    #finding new centroid
    new_centroid_cluster_list = []
    for cluster in cluster_list:        
        new_centroid = calc_mean(cluster)
        new_centroid_cluster_list.append(new_centroid)
    
    
    
    # for node in new_centroid_cluster_list:
    #     print('nccl: ',node)

    for index in range(k):
        if new_centroid_cluster_list[index] == centroid_cluster_list[index]:
            flag = 1
        else:
            flag = 0
            break
            
    centroid_cluster_list = [c for c in new_centroid_cluster_list]
    # print("new iteration")
    

for node in cluster_list:
        print('cl: ',node)





#  *********  T2  Bigram ****************


# print(cluster_list)

#function to calculate centroid
def calc_centroid(cluster):
    sum = np.zeros(no_of_rows)
    for x in cluster:
        sum += np.array(transposed_matrix[x])

    if len(cluster)>0:
        centroid_vector = sum/len(cluster)
    else:
        centroid_vector = sum
    return (centroid_vector.tolist())

# Function to convert a list of bigrams to sentences
def bigrams_to_sentences(bigram_list):
    sentences = []
    for bigram in bigram_list:
        words = bigram.split()  # Split the bigram into individual words
        sentence = (words[0])  # Join the words to form a sentence
        sentences.append(sentence)
    return sentences

#finding centroid of the each cluster
centroid_all_cluster = []
for cluster in cluster_list:
    centroid = calc_centroid(cluster)
    centroid_all_cluster.append(centroid)

    
def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Both points must have the same number of dimensions")
        
    squared_distance = sum([(x - y) ** 2 for x, y in zip(point1, point2)])
    distance = math.sqrt(squared_distance)
    return distance

cluster_sentence = []
cluster_sentence_index = []
cluster_all_sentences = []
index = -1
for cluster in cluster_list:
    index += 1
    min_distance = 100000
    sentence_index = -1
    all_sentences =[]
    for node in cluster:
        all_sentences.append(sentence_list[node])
        distance = euclidean_distance(centroid_all_cluster[index], transposed_matrix[node])
        if distance < min_distance:
            min_distance = distance
            sentence_index = node 
    cluster_all_sentences.append(all_sentences)
    cluster_sentence_index.append(sentence_index)
    cluster_sentence.append(sentence_list[sentence_index])


cluster_all_bigram = []
for cluster in cluster_all_sentences:
    cluster_bigram = []
    for sentence in cluster:
        bigrams = []
        for index in range(len(sentence)-1):
            bigram = sentence[index] + ' ' + sentence[index + 1]
            bigrams.append(bigram)
        cluster_bigram.append(bigrams)
    cluster_all_bigram.append(cluster_bigram)

cluster_bigram = []
for sentence in cluster_sentence:
    bigrams = []
    for index in range(len(sentence)-1):
        bigram = sentence[index] + ' ' + sentence[index + 1]
        bigrams.append(bigram)
    cluster_bigram.append(bigrams)


secondary_bigram_cluster_of_cluster = []
secondary_sentence_cluster_of_cluster = []

index = -1
for cluster1 in cluster_bigram:
    index += 1
    cluster = []
    sent_cluster = []
    clusterc = cluster_all_bigram[index]
    flag = 0
    for cluster2 in clusterc:
        if flag == 0:
            if cluster1 == cluster2:
                continue
            else:
                count = 0
                for word in cluster1:
                    if word in cluster2:
                        count += 1
                if count >= 0:
                    flag = 1
                    cluster.append(cluster2) 
                    sent_cluster.append(bigrams_to_sentences(cluster2))
    secondary_bigram_cluster_of_cluster.append(cluster)
    secondary_sentence_cluster_of_cluster.append(sent_cluster)


def construct_sentence_graph(S1, S2):
    # Create a directed graph
    G = nx.DiGraph()

    # Dummy start and end nodes
    dummy_start = "START"
    dummy_end = "END"

    # Add dummy nodes to the graph
    G.add_node(dummy_start)
    G.add_node(dummy_end)

    # Create bigrams and add nodes and edges for S1
    S1_bigrams = S1.split()
    for i, bigram in enumerate(S1_bigrams):
        G.add_node(bigram)
        if i == 0:
            # Add an incoming edge from the dummy start node to the first bigram
            G.add_edge(dummy_start, bigram)
        elif i == len(S1_bigrams) - 1:
            # Add an outbound edge from the last bigram to the dummy end node
            G.add_edge(bigram, dummy_end)
        else:
            # Add an incoming edge from the previous bigram to the current bigram
            G.add_edge(S1_bigrams[i - 1], bigram)

    # Create bigrams and add nodes and edges for S2
    S2_bigrams = S2.split()
    for i, bigram in enumerate(S2_bigrams):
        G.add_node(bigram)
        if i == 0:
            # Add an incoming edge from the dummy start node to the first bigram
            G.add_edge(dummy_start, bigram)
        elif i == len(S2_bigrams) - 1:
            # Add an outbound edge from the last bigram to the dummy end node
            G.add_edge(bigram, dummy_end)
        else:
            # Add an incoming edge from the previous bigram to the current bigram
            G.add_edge(S2_bigrams[i - 1], bigram)

    return G

def generate_sentence(graph):
    # Initialize the sentence with the dummy start node
    sentence = [""]
    
    # Traverse the graph to generate the sentence
    current_node = "START"
    while current_node != "END":
        neighbors = graph.neighbors(current_node)
        next_neighbors = [neighbor for neighbor in neighbors]
        if next_neighbors:
            next_node = random.choice(next_neighbors)
            sentence.append(next_node)
            current_node = next_node
        else:
            # Break if there are no more neighbors
            break

    # Check if "END" is in the sentence before attempting to remove it
    if "END" in sentence:
        sentence.remove("END")

    # Join the words to form the final sentence
    final_sentence = ' '.join(sentence)

    # Split the sentence into words
    words = final_sentence.split()    
    # Get the last word
    last_word = words[-1]

    
    return final_sentence


final_sentence_cluster_of_cluster = []
index = -1
for list in secondary_sentence_cluster_of_cluster:
    index += 1
    dummy = []
    if len(list) == 0:
        final_sentence_cluster_of_cluster.append(cluster_sentence[index])
    else:        
        # Convert the list into a sentence
        sentence1 = ' '.join(cluster_sentence[index])
        sentence2 = ' '.join(list[0])

        graph = construct_sentence_graph(sentence1, sentence2)
        generated_sentence = generate_sentence(graph)
        # print("Generated Sentence:", generated_sentence)
        generated_sentence = generated_sentence.split()
        final_sentence_cluster_of_cluster.append(generated_sentence)



# ********************      task3           *************************
 
file = open("‘Summary_SentenceGraph.txt", "w")

for ind in sorted(cluster_sentence_index):
    print(sentence_list[sorted_nodes[ind]])
    file.write(" ".join(sentence_list[sorted_nodes[ind]]) + "\n")

file.close()
    