import os
import argparse
import networkx as nx
from gensim.models import Word2Vec
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd

# Use: /usr/bin/python embeddings/DeepWalk/DeepWalk.py /thayerfs/home/f006dg0/Learning_Embeddings/SNAP/datasets/email-Eu-core-temporal.txt

def read_directed_graph(file_path):
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=['source', 'target'])

    # Create a directed graph from the DataFrame
    G = nx.from_pandas_edgelist(df, source='source', target='target', create_using=nx.DiGraph)

    return G

def get_random_walk(graph, node, n_steps=4):
    """Generate a random walk starting from the given node."""
    local_path = [str(node)]
    target_node = node
    for _ in range(n_steps):
        neighbors = list(graph.neighbors(target_node))
        if neighbors:
            target_node = random.choice(neighbors)
            local_path.append(str(target_node))
        else:
            break
    return local_path

def generate_walks(graph, num_walks=10, walk_length=4):
    """Generate random walks in the graph."""
    walk_paths = []
    for node in graph.nodes():
        for _ in range(num_walks):
            walk_paths.append(get_random_walk(graph, node, walk_length))
    return walk_paths

def email(success, dimensions, output_file, error_type=None):
    # Email configurations
    sender_email = "felicia.schenkelberg.th@dartmouth.edu"
    receiver_email = "felicia.schenkelberg.th@dartmouth.edu"

    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email

    if success:
        subject = "DeepWalk Embeddings Generation Complete"
        message = f"Graph embeddings for dimension {dimensions} saved to {output_file}"
    else:
        if error_type == "MemoryError":
            subject = "MemoryError Encountered in DeepWalk Script"
            message = f"A MemoryError occurred while generating graph embeddings for dimension {dimensions}."
        
        elif error_type == "SystemError":
            subject = "SystemError Encountered in DeepWalk Script"
            message = f"A SystemError occurred while generating graph embeddings for dimension {dimensions}."
        else:
            subject = "Error Generating DeepWalk Embeddings"
            message = f"An error occurred while generating graph embeddings for dimension {dimensions}."

    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Connect to the SMTP server
    with smtplib.SMTP('smtp.dartmouth.edu', 25) as server:
        # Send the email
        server.sendmail(sender_email, receiver_email, msg.as_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file path and save path.")
    parser.add_argument("file_path", help="Path to the input file")
    args = parser.parse_args()

    filename, _ = os.path.splitext(args.file_path)
    save_path = '/thayerfs/home/f006dg0/embeddings/DeepWalk/'

    # Read graph and obtain node attributes and labels
    G = read_directed_graph(args.file_path)

    # Define dimensions
    dimension_list = [32, 128, 256]

    for dimensions in dimension_list:
        # Output file names
        embedding_output_file = f"{filename}_dim_{dimensions}.txt"

        # Generate random walks
        walk_paths = generate_walks(G)

        try:
            # Train Word2Vec model
            model = Word2Vec(sentences=walk_paths, vector_size=dimensions, window=5, min_count=1, sg=1, hs=0, negative=10, epochs=10)
            
            # Save embeddings
            model.wv.save_word2vec_format(os.path.join(save_path, embedding_output_file))
            
            # Send success email
            email(True, dimensions, embedding_output_file)

        except MemoryError as mem_error:
            # Send memory error email
            email(False, dimensions, embedding_output_file, error_type="MemoryError")
            print("MemoryError occurred. Email notification sent.")

        except SystemError as sys_error:
            # Send system error email
            email(False, dimensions, embedding_output_file, error_type="SystemError")
            print("SystemError occurred. Email notification sent.")

        except Exception as e:
            # Send error email
            email(False, dimensions, embedding_output_file)

    print("Done.")
