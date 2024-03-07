import os
import torch as pt
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

def read_data_and_count(file_path):
    max_shared_elements = 0
    data_list = []

    with open(file_path, 'r') as file:
        for line in file:
            entries = line.strip().split(' ')
            count = len(entries) - 1
            if count > max_shared_elements:
                max_shared_elements = count
            float_values = [float(entry) for entry in entries[1:]]
            data_list.append(float_values)

    filtered_data_list = [data for data in data_list if len(data) == max_shared_elements]
    tensor_data = pt.tensor(filtered_data_list)

    # Normalize data using Min-Max scaling
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(tensor_data.numpy())

    return pt.tensor(normalized_data)

def email(success, dimensions, graph, output_file):
    # Email configurations
    sender_email = "felicia.schenkelberg.th@dartmouth.edu"
    receiver_email = "felicia.schenkelberg.th@dartmouth.edu"

    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email

    if success:
        subject = "Graph Generation Complete"
        message = f"{graph} graph for dimension {dimensions} saved to {output_file}"
    else:
        subject = "Error Generating Graph"
        message = f"Error generating {graph} graph for dimension {dimensions}"

    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Connect to the SMTP server
    with smtplib.SMTP('smtp.dartmouth.edu', 25) as server:
        server.sendmail(sender_email, receiver_email, msg.as_string())

def filename(data_file, graph):
    # Extract the base filename from the provided path
    base = os.path.basename(data_file)
    root, _ = os.path.splitext(base)

    # Create the new filename with extension
    if graph == "elbow":
        filename = f"{root}_elbow.png"
    elif graph == "silhouette":
        filename = f"{root}_silhouette.png"
    return filename

def calculate_inertia(X, max_clusters):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias

def elbow(X_pt, data_dir, data_file, max_clusters=41):
    inertias = calculate_inertia(X_pt.numpy(), max_clusters)

    # Elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.xticks(np.arange(1, max_clusters + 1, 1))
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, filename(data_file, "elbow")))
    plt.close()

def calculate_silhouette(X, max_clusters):
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

def silhouette(X_pt, data_dir, data_file, max_clusters=41):
    silhouette_scores = calculate_silhouette(X_pt.numpy(), max_clusters)

    # Silhouette plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.xticks(np.arange(2, max_clusters + 1, 1))
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, filename(data_file, "silhouette")))
    plt.close()

def main() -> None:
    dimensions = [32, 64, 128, 256]
    data_dir = '/scratch/f006dg0/mcas-gmra/pymm-gmra/experiments/SNAP/reduced_embeddings/{}/'
    file_paths = [
        'email-Eu-core-temporal_dim_{}_dram.txt',
    ]

    for dim in dimensions:
        for file_path in file_paths:
            data_file = os.path.join(data_dir.format(dim), file_path.format(dim))
            
            print(f"Loading data for dimension {dim} from {data_file}")
            X_pt = read_data_and_count(data_file)
            print("Done loading data")

            try:
                # Generate elbow graph
                elbow(X_pt, data_dir.format(dim), data_file)

                # Send success email
                email(True, dim, "elbow", os.path.join(data_dir.format(dim), filename(data_file, "elbow")))
            
            except Exception as e:
                # Send error email
                email(False, dim, "elbow", data_file)
                print(f"Error processing dimension {dim} for file {data_file}: {e}")
            
            try:
                # Generate silhouette graph
                silhouette(X_pt, data_dir.format(dim), data_file)

                # Send success email
                email(True, dim, "silhouette", os.path.join(data_dir.format(dim), filename(data_file, "silhouette")))

            except Exception as e:
                # Send error email
                email(False, dim, "silhouette", data_file)
                print(f"Error processing dimension {dim} for file {data_file}: {e}")

if __name__ == "__main__":
    main()
