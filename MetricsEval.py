import os
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Step 1: Parse the Data
def load_friends(data_dir):
    friends_dict = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            username = filename[:-4]
            with open(os.path.join(data_dir, filename), 'r') as file:
                friends = file.read().splitlines()
                friends_dict[username] = friends
    return friends_dict

data_dir = "data"
friends_dict = load_friends(data_dir)

# Step 2: Create a Graph
def create_graph(friends_dict):
    G = nx.Graph()
    for user, friends in friends_dict.items():
        for friend in friends:
            G.add_edge(user, friend)
    return G

G = create_graph(friends_dict)

# Step 3: Prepare Data for Link Prediction
def generate_samples(G, num_samples):
    positive_samples = list(G.edges())
    non_edges = list(nx.non_edges(G))
    negative_samples = random.sample(non_edges, num_samples)
    return positive_samples, negative_samples

num_samples = len(G.edges())
positive_samples, negative_samples = generate_samples(G, num_samples)

# Step 4: Compute Features
def compute_features(G, samples):
    features = {
        'jaccard': [],
        'adamic_adar': [],
        'preferential_attachment': [],
        'resource_allocation': []
    }
    for u, v in samples:
        # Jaccard coefficient
        jaccard_coeff = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
        features['jaccard'].append(jaccard_coeff)
        
        # Adamic-Adar index
        adamic_adar_index = list(nx.adamic_adar_index(G, [(u, v)]))[0][2]
        features['adamic_adar'].append(adamic_adar_index)
        
        # Preferential attachment
        preferential_attachment = list(nx.preferential_attachment(G, [(u, v)]))[0][2]
        features['preferential_attachment'].append(preferential_attachment)
        
        # Resource Allocation index
        resource_allocation_index = list(nx.resource_allocation_index(G, [(u, v)]))[0][2]
        features['resource_allocation'].append(resource_allocation_index)
        
    return features

features = compute_features(G, positive_samples + negative_samples)

# Step 5: Evaluate Individual Features
def evaluate_feature(feature_values, labels):
    X_train, X_test, y_train, y_test = train_test_split(feature_values, labels, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    return accuracy, auc

# Create labels
labels = [1] * len(positive_samples) + [0] * len(negative_samples)

# Evaluate each feature
results = {}
for metric, values in features.items():
    accuracy, auc = evaluate_feature([[val] for val in values], labels)
    results[metric] = (accuracy, auc)

# Print results
for metric, (accuracy, auc) in results.items():
    print(f"Metric: {metric}")
    print(f"Accuracy: {accuracy}")
    print(f"AUC Score: {auc}")
    print()

# Plot the results
metrics = list(results.keys())
accuracies = [result[0] for result in results.values()]
aucs = [result[1] for result in results.values()]

x = range(len(metrics))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(x, accuracies, align='center', alpha=0.7)
plt.xticks(x, metrics)
plt.xlabel('Metric')
plt.ylabel('Accuracy')
plt.title('Accuracy by Metric')

plt.subplot(1, 2, 2)
plt.bar(x, aucs, align='center', alpha=0.7)
plt.xticks(x, metrics)
plt.xlabel('Metric')
plt.ylabel('AUC Score')
plt.title('AUC Score by Metric')

plt.tight_layout()
plt.show()