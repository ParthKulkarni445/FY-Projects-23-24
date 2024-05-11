import csv  # Importing the CSV module to read and write CSV files
import networkx as nx  # Importing NetworkX for graph operations
import random  # Importing the random module for generating random numbers
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import numpy as np  # Importing NumPy for numerical operations
import pandas as pd  # Importing pandas for data manipulation
from sklearn.linear_model import LinearRegression  # Importing Linear Regression from scikit-learn

# Creating a function to read the data from the CSV file
def input_data():
    # Create an empty list to store the data
    l=[]
    # Open the CSV file in read mode
    with open('Project_2_dataset.csv', 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Iterate through each row in the CSV file
        for row in reader:
            # Append each row to the list
            l.append(row)      
    # Return the list containing the data from the CSV file
    return l

# Creating a dictionary of nodes and their respective outlinks
def create_node_dict():
    # Create an empty dictionary to store nodes and their outlinks
    node_dict = {}
    # Iterate through each row of data obtained from input_data() function
    for i in input_data():
        # The first element of the row is the node, and the rest are its outlinks
        node_dict[i[0]] = i[1:]
    # Return the dictionary containing nodes and their outlinks
    return node_dict

# Create a function to generate a random walk
def walk_random(G):
    # Initialize a dictionary to store the visit count of each node
    walk_points = {}
    for i in G.nodes():
        walk_points[i] = 0
    # Get the list of nodes in the graph
    nodelist = list(G.nodes())
    # Choose a random starting node
    r = random.choice(nodelist)
    # Increment the visit count of the starting node
    walk_points[r] += 1
    # Get the outlinks of the starting node
    outlinks = list(G.out_edges(r))
    # Set a counter to limit the number of steps in the random walk
    ct = 0
    while ct != 1e6:  # Limit the walk to 1 million steps
        # If the current node has no outlinks, choose a random node as the next focus
        if len(outlinks) == 0:
            focus = random.choice(nodelist)
        else:
            # Choose a random outlink as the next focus
            focus = random.choice(outlinks)[1]
        # Increment the visit count of the next focus node
        walk_points[focus] += 1
        # Get the outlinks of the next focus node
        outlinks = list(G.out_edges(focus))
        # Increment the step counter
        ct += 1
    
    return walk_points

# Function to predict whether a link can exist between two nodes
def check_link(adj_matrix, weights, node1, node2):
    # Extract the column corresponding to node2 from the adjacency matrix
    target_col = np.array(adj_matrix)[:, node2]
    # Remove the element corresponding to node1 from the target column
    target_col = np.delete(target_col, node1)
    # Convert the weights list to a numpy array
    weights = np.array(weights)
    # Check if there are no weights associated with node1
    if np.all(weights[node1] == 0):
        # If the rank of node2 is greater than or equal to the average rank, return a positive prediction
        if ranks[nodelist[node2]] >= avg_rank:
            return random.choice([-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # If the rank of node2 is less than the average rank, return a negative prediction
        else:
            return random.choice([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1])
    else:
        # Predict the link by taking the dot product of the weights of node1 and the target column
        pred = np.dot(weights[node1], target_col)
        return pred

# Function to apply check_link to all pairs of nodes and return count in cases where link does not exist but our model predicts it would have
def new_links(adj_matrix):
    print("Please wait, while we predict the new links...")
    # Initialize a list to store the prediction results
    l = [[[] for i in range(len(nodelist))] for j in range(len(nodelist))]
    # Initialize a counter for the predicted new links
    ct = 0
    # Iterate through each pair of nodes in the adjacency matrix
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            # If there is already a link between the nodes
            if adj_matrix[i][j] == 1:
                # Set the corresponding entry in the result list to 1
                l[i][j] = 1
            # If there is no link between the nodes
            elif i != j and adj_matrix[i][j] == 0:
                # Predict whether a link could exist between the nodes
                if check_link(adj_matrix, lr_weights, i, j) >= 0:
                    # If the model predicts a link, set the corresponding entry to 1 and increment the counter
                    l[i][j] = 1
                    ct += 1
                else:
                    # If the model predicts no link, set the corresponding entry to 0
                    l[i][j] = 0
            else:
                # If the nodes are the same or the entry is already 1, set the entry to 0
                l[i][j] = 0
    # Print the number of new edges predicted and the percentage of total possible edges
    print("The no. of new edges predicted:", ct, " i.e the graph now contains", ((ct + len(G.edges())) * 100) / (143 * 142), "% of the total possible edges", end='\n\n')
    return l

# Create a function to check whether the graph exhibits homophily
def check_homophily(adj_matrix, cs_nodes, mnc_nodes):
    # Initialize a counter for the number of edges exhibiting homophily
    count = 0
    # Iterate through each pair of nodes in the adjacency matrix
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            # If there is an edge between the nodes and they belong to different communities
            if adj_matrix[i][j] == 1 and ((nodelist[i] in cs_nodes and nodelist[j] in mnc_nodes) or (nodelist[j] in cs_nodes and nodelist[i] in mnc_nodes)):
                # Increment the counter
                count += 1
    # Calculate the homophily coefficient
    homophily_coefficient = 1 - 2 * count / len(G.edges())
    # Determine whether homophily or heterophily exists based on the coefficient
    if homophily_coefficient > 0:
        return count, "Homophily exists"
    else:
        return count, "Homophily does not exist, instead heterophily exists"

# Main code
if __name__ == "__main__":
    # Creating the graph
    node_dict = create_node_dict()
    G = nx.DiGraph()
    G.add_nodes_from(node_dict.keys())
    for k,v in node_dict.items():
        for i in v:
            if(i!=''):
                G.add_edge(k,i)
    
    # Calculate pageranks by random walk method
    walk_points = walk_random(G)
    pr_random_walk = sorted(walk_points.items(), key=lambda x: x[1], reverse=True)

    # Print the leader of the class by random walk method
    print("The leader of the class by random walk method is:",pr_random_walk[0][0],end='\n\n')

    # Finding pageranks using built-in function to verify the random walk results
    ranks = nx.pagerank(G)
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    # Find the average pagerank
    avg_rank = sum(ranks.values())/len(ranks)

    # Create adjacency matrix of the graph
    nodelist = list(G.nodes())
    adjmat = [[0 for i in range(len(nodelist))] for j in range(len(nodelist))]
    for i in range(len(nodelist)):
        for j in range(len(nodelist)):
            if(G.has_edge(nodelist[i],nodelist[j])):
                adjmat[i][j] = 1

    # Initialize an empty list to store the weights obtained from linear regression
    lr_weights = []
    # Create a copy of the adjacency matrix
    adjmatcp = adjmat.copy()
    # Convert the adjacency matrix to a numpy array
    mat = np.array(adjmatcp)
    # Iterate through each row of the adjacency matrix
    for i in range(len(mat)):
        # Extract the target row for regression
        target_row = mat[i] 
        # Create a copy of the matrix excluding the target row
        other_rows = mat.copy()
        other_rows = np.delete(other_rows, (i), axis=0)
        # Transpose the matrix to match the dimensions for linear regression
        other_rows = other_rows.T
        # Fit a linear regression model using other rows as features and target_row as target
        model = LinearRegression().fit(other_rows, target_row)
        # Append the coefficients obtained from the model to lr_weights list
        lr_weights.append(model.coef_)

    # Predict new links
    new_mat = new_links(adjmat)
    new_mat_df = pd.DataFrame(new_mat, index=nodelist, columns=nodelist)
    print("The new adjancency matrix dataframe:",new_mat_df,sep='\n',end='\n\n')

    # Make a list of nodes belonging to students from CSE branch
    cs_nodes=[]
    for i in range(len(nodelist)):
        if('CSB' in nodelist[i]):
            cs_nodes.append(nodelist[i])

    # Make a list of nodes belonging to students from MNC branch
    mnc_nodes=[]
    for i in range(len(nodelist)):
        if('MCB' in nodelist[i]):
            mnc_nodes.append(nodelist[i])

    # Print the number of cross edges between CSE and MNC and whether homophily or heterophily exists
    print("The number of cross edges between CSE and MNC is: ",check_homophily(adjmat, cs_nodes, mnc_nodes)[0], " and",check_homophily(adjmat, cs_nodes, mnc_nodes)[1])
