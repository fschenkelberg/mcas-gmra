import argparse
import os
import torch as pt
from sklearn.cluster import KMeans
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import matplotlib.pyplot as plt

# Read Data
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

    return tensor_data

def email(success, dimensions, output_file, error_type=None):
    # Email configurations
    sender_email = "felicia.schenkelberg.th@dartmouth.edu"
    receiver_email = "felicia.schenkelberg.th@dartmouth.edu"

    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email

    if success:
        subject = "K-Means Generation Complete"
        message = f"Cluster Graph for dimension {dimensions} saved to {output_file}"
    
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Connect to the SMTP server
    with smtplib.SMTP('smtp.dartmouth.edu', 25) as server:
        server.sendmail(sender_email, receiver_email, msg.as_string())

def filename(data_file):
    # Extract the base filename from the provided path
    base = os.path.basename(data_file)
    root, _ = os.path.splitext(base)
    # Create the new filename with extension
    filename = f"{root}_kmeans.png"

    return filename

class KMeansModel:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit(self, X):
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(X)

    def predict(self, X):
        return self.kmeans.predict(X)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="directory to where results will save to")
    parser.add_argument("--data_file", type=str, help="path to the data file")
    parser.add_argument("--n_clusters", type=int, default=2, help="number of clusters")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("loading data")
    X_pt = read_data_and_count(args.data_file)
    print("done")

    kmeans_model = KMeansModel(args.n_clusters)
    kmeans_model.fit(X_pt.numpy())
    # labels = kmeans_model.predict(X_pt.numpy())

    # Save results
    results_image = os.path.join(args.data_dir, filename(args.data_file))
    """
    results_file = os.path.join(args.data_dir, filename(args.data_file))
    with open(results_file, 'w') as file:
        file.write("Cluster Assignments:\n")
        for i, label in enumerate(labels):
            file.write(f"Vector {i+1}: Cluster {label}\n")
    """

    # print("saving results to [%s]" % results_file)
    email(True, args.n_clusters, results_image)

if __name__ == "__main__":
    main()
