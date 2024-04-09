import networkx as nx
from node2vec import Node2Vec
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import argparse
import os
import pandas as pd
from node2vec.edges import HadamardEmbedder

# Use: /usr/bin/python Node2Vec.py /thayerfs/home/f006dg0/email-Eu-core.txt

def read_directed_graph(file_path):
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, comment='#', names=['source', 'target'])

    # Create a directed graph from the DataFrame
    G = nx.from_pandas_edgelist(df, source='source', target='target', create_using=nx.DiGraph)

    return G

def email(success, dimensions, output_file, error_type=None):
    # Email configurations
    sender_email = "felicia.schenkelberg.th@dartmouth.edu"
    receiver_email = "felicia.schenkelberg.th@dartmouth.edu"

    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email

    if success:
        subject = "Node2Vec Embeddings Generation Complete"
        message = f"Graph embeddings for dimension {dimensions} saved to {output_file}"
    else:
        if error_type == "MemoryError":
            subject = "MemoryError Encountered in Node2Vec Script"
            message = f"A MemoryError occurred while generating graph embeddings for dimension {dimensions}."
        
        elif error_type == "SystemError":
            subject = "SystemError Encountered in Node2Vec Script"
            message = f"A SystemError occurred while generating graph embeddings for dimension {dimensions}."
        else:
            subject = "Error Generating Node2Vec Embeddings"
            message = f"An error occurred while generating graph embeddings for dimension {dimensions}. Error Message: {error_type}"

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

    filename, extension = os.path.splitext(args.file_path)
    save_path = '/scratch/f006dg0/mcas-gmra/pymm-gmra/experiments/SNAP/learned_embeddings/Edge2Vec/'

    # Read graph and obtain node attributes and labels
    G = read_directed_graph(args.file_path)

    # Define dimensions
    dimension_list = [32, 64, 128, 256]

    for dimensions in dimension_list:

        # Output file names
        embedding_output_file = f"{filename}_dim_{dimensions}.txt"

        #node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4)
        #model = node2vec.fit(window=10, min_count=1, batch_words=4, vector_size=dimensions)
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=100, workers=2)  # Reduced num_walks and workers
        model = node2vec.fit(window=10, min_count=1, batch_words=2, vector_size=dimensions)  # Reduced batch_words

        try:
            edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
            edges_kv = edges_embs.as_keyed_vectors()
            edges_kv.save_word2vec_format(embedding_output_file)

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
            email(False, dimensions, embedding_output_file, error_type=e)
        
    print("Done.")
