import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.externals import joblib
import torch
import torch.nn as nn

# Initialize the main window
root = tk.Tk()
root.title("K-means Clustering")

# Create data entry fields
data_frame = ttk.LabelFrame(root, text="Enter Data")
data_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

data_label = ttk.Label(data_frame, text="Enter data points (comma separated):")
data_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
data_entry = ttk.Entry(data_frame, width=50)
data_entry.grid(row=0, column=1, padx=10, pady=5)

k_label = ttk.Label(data_frame, text="Enter the number of clusters (K):")
k_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
k_entry = ttk.Entry(data_frame, width=5)
k_entry.grid(row=1, column=1, padx=10, pady=5)

# Function to perform K-means clustering
def perform_clustering():
    data_str = data_entry.get()
    k_value = k_entry.get()

    try:
        data = [list(map(float, point.split(",")) ) for point in data_str.split(";")]
        k_value = int(k_value)

        if k_value <= 0:
            messagebox.showerror("Error", "Number of clusters (K) should be greater than 0.")
        else:
            kmeans = KMeans(n_clusters=k_value)
            kmeans.fit(data)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # Display the clustered data using Seaborn
            data_df = pd.DataFrame(data, columns=["X", "Y"])
            data_df["Cluster"] = labels
            sns.lmplot(x="X", y="Y", data=data_df, fit_reg=False, hue="Cluster", markers=["o", "s", "D", "v"])
            plt.title("K-means Clustering")
            
            # Plot the centroids
            plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label="Centroids")
            
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend(loc='best')
            plt.show()

    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter valid data and K value.")

# Create the cluster button
cluster_button = ttk.Button(root, text="Cluster Data", command=perform_clustering)
cluster_button.grid(row=0, column=1, padx=10, pady=10)

# Function to generate random data for clustering
def generate_random_data():
    n_samples = 100
    n_features = 2
    centers = 4
    data, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)
    data_str = ""
    for point in data:
        data_str += f"{point[0]},{point[1]};"
    data_entry.delete(0, tk.END)
    data_entry.insert(0, data_str[:-1])
    k_entry.delete(0, tk.END)
    k_entry.insert(0, centers)

# Create the generate random data button
random_data_button = ttk.Button(root, text="Generate Random Data", command=generate_random_data)
random_data_button.grid(row=0, column=2, padx=10, pady=10)

# Function to save the model
def save_model():
    kmeans = KMeans(n_clusters=int(k_entry.get()))
    data_str = data_entry.get()
    data = [list(map(float, point.split(",")) ) for point in data_str.split(";")]
    kmeans.fit(data)
    joblib.dump(kmeans, "kmeans_model.pkl")
    messagebox.showinfo("Saved Model", "The K-means model has been saved as 'kmeans_model.pkl'.")

# Create the save model button
save_model_button = ttk.Button(root, text="Save Model", command=save_model)
save_model_button.grid(row=0, column=3, padx=10, pady=10)

# Function to load the model
def load_model():
    try:
        kmeans = joblib.load("kmeans_model.pkl")
        data_str = data_entry.get()
        data = [list(map(float, point.split(",")) ) for point in data_str.split(";")]
        labels = kmeans.predict(data)
        perform_clustering()  # Re-run clustering
        messagebox.showinfo("Loaded Model", "The K-means model has been loaded.")
    except FileNotFoundError:
        messagebox.showerror("Error", "No saved model found. Save a model first.")

# Create the load model button
load_model_button = ttk.Button(root, text="Load Model", command=load_model)
load_model_button.grid(row=0, column=4, padx=10, pady=10)

# Start the GUI main loop
root.mainloop()
