import os
import sys
import time
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import torch
import lightning as L
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

FILE = Path(__file__).resolve()
SCRIPTS = FILE.parents[0]
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from vae import VAE
from utils import Parser
from argparse import ArgumentParser

def latent_space(augmentations: list, vae: VAE):
    # Create dataset for the latent space
    dataset_ids = []
    latent_vectors = []
    labels = []
    pred_labels = []
    kld_loss = []
    mse_loss = []
    class_loss = []

    for id, transformation in enumerate(augmentations):
        print(f"Processing dataset {id}...")
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformation)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Iterate through the dataset
        dataset_id = torch.ones(1) * id
        with torch.no_grad():
            for data, target in tqdm(loader, leave=True):
                # Run the encoder part
                mu, log_var = vae.encode(data)
                z = vae.reparameterize(mu, log_var)
                pred = vae.classifier(z)
                recon = vae.decode(z)
                mse, kld = vae.loss_function(recon, data, mu, log_var)

                # Append latent vectors and labels to the lists
                latent_vectors.extend(mu.cpu().numpy())
                labels.extend(target.cpu().numpy())
                dataset_ids.append(dataset_id.cpu().numpy())
                pred_labels.append(torch.argmax(pred).cpu().numpy())
                kld_loss.append(kld.cpu().numpy())
                mse_loss.append(mse.cpu().numpy())
                class_loss.append(F.cross_entropy(pred, target).cpu().numpy())

    # Convert lists to numpy arrays
    latent_vectors = np.array(latent_vectors, dtype=np.float32)
    labels = np.array(labels)
    dataset_ids = np.array(dataset_ids)
    pred_labels = np.array(pred_labels)
    kld_loss = np.array(kld_loss)
    mse_loss = np.array(mse_loss)
    class_loss = np.array(class_loss)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, )
    start_time = time.time()
    latent_vectors_2d = tsne.fit_transform(latent_vectors)
    total_time = time.time() - start_time
    print(f"t-SNE done! ETA: {total_time}")

    data = {
        'dataset_id': dataset_ids.flatten(),
        'labels': labels,
        'pred_labels': pred_labels,
        'kld_loss': kld_loss,
        'mse_loss': mse_loss,
        'class_loss': class_loss,
        'latent_vec_2d_x': latent_vectors_2d[:, 0],
        'latent_vec_2d_y': latent_vectors_2d[:, 1]
    }
    df = pd.DataFrame(data)

    # Grouping by dataset
    part_size = len(augmentations)
    result = [latent_vectors[i*int(len(latent_vectors)/part_size):(i+1)*int(len(latent_vectors)/part_size)] for i in range(0, part_size)]

    return df, result

def plot_latent_space(df: pd.DataFrame):
    now = datetime.now()
    formatted_date = now.strftime('%Y%m%d%H%M%S')
    output_dir = f'output/augmentation_robustness/{formatted_date}/'
    os.makedirs(output_dir, exist_ok=True)

    max_id = df['dataset_id'].max()
    ref = df[df['dataset_id'] == 0.0]

    attributes = ['dataset_id', 'labels', 'pred_labels', 'kld_loss', 'mse_loss', 'class_loss']
    for attr in attributes:
        for i in tqdm(range(int(max_id+1)), desc=f"Plotting by {attr}"):
            filtered = df[df['dataset_id'] == i]
            plt.figure(figsize=(10, 6))
            plt.scatter(ref['latent_vec_2d_x'], ref['latent_vec_2d_y'], c=filtered[attr], cmap='viridis')
            plt.scatter(filtered['latent_vec_2d_x'], filtered['latent_vec_2d_y'], c=filtered[attr], cmap='viridis')
            plt.title(f't-SNE Visualization of Latent Vectors by {attr}')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.colorbar(label=f'{attr}')
            plt.grid(True)
            plt.savefig(f"output/augmentation_robustness/{formatted_date}/lspace_{i}_{attr}.png",transparent=True)
            plt.clf()

def stat_latent_space(latent_spaces: list, df: pd.DataFrame):
    for i, lspace in enumerate(latent_spaces):
        print(f"Dataset {i}")
        # Average and std deviation of the latent space vectors size
        latent_length = np.linalg.norm(lspace, axis=1)
        avg_length = np.mean(latent_length)
        std_length = np.std(latent_length)
        print(f"[01/18] Average latent space vector size: {avg_length}")
        print(f"[02/18] Std deviation of latent space vector size: {std_length}")

        # Average and std deviation of the latent space vectors size by class
        avgs = []
        stds = []
        for c in range(10):
            class_indices = df.index[df['labels'] == c].to_list()
            class_indices = [i % 10000 for i in class_indices]
            class_latent = lspace[class_indices]
            class_latent_length = np.linalg.norm(class_latent, axis=1)
            avgs.append(np.mean(class_latent_length))
            stds.append(np.std(class_latent_length))
        print(f"[03/18] Average latent space vector size by class: {avgs}")
        print(f"[04/18] Std  of latent space vector size by class: {stds}")

        #Average and std deviation of the diff vectors size
        diff = lspace - latent_spaces[0]
        diff_length = np.linalg.norm(diff, axis=1)
        avg_diff_length = np.mean(diff_length)
        std_diff_length = np.std(diff_length)
        print(f"[05/18] Average diff vector size: {avg_diff_length}")
        print(f"[06/18] Std deviation of diff vector size: {std_diff_length}")

        # Average and std deviation of the diff vectors size by class
        avgs = []
        stds = []
        for c in range(10):
            class_indices = df.index[df['labels'] == c].to_list()
            class_indices = [i % 10000 for i in class_indices]
            class_diff = diff[class_indices]
            class_diff_length = np.linalg.norm(class_diff, axis=1)
            avgs.append(np.mean(class_diff_length))
            stds.append(np.std(class_diff_length))

        print(f"[07/18] Average diff vector size by class: {avgs}")
        print(f"[08/18] Std  of diff vector size by class: {stds}")

        # Average and std deviation of the cosine similarity of diff vectors
        cos_sim = cosine_similarity(lspace, latent_spaces[0])
        avg_cos_sim = np.mean(cos_sim)
        std_cos_sim = np.std(cos_sim)
        print(f"[09/18] Average cosine similarity of diff vectors: {avg_cos_sim}")
        print(f"[10/18] Std dev cosine similarity of diff vectors: {std_cos_sim}")

        # Average and std deviation of the cosine similarity of diff vectors by class
        avgs = []
        stds = []
        for c in range(10):
            class_indices = df.index[df['labels'] == c].to_list()
            class_indices = [i % 10000 for i in class_indices]
            class_cos_sim = cosine_similarity(lspace[class_indices], latent_spaces[0][class_indices])
            avgs.append(np.mean(class_cos_sim))
            stds.append(np.std(class_cos_sim))

        print(f"[11/18] Average cosine similarity of diff vectors by class: {avgs}")
        print(f"[12/18] Std dev cosine similarity of diff vectors by class: {stds}")

        # Average and std deviation of the latent space KLD, MSE and class loss
        avg_kld = df['kld_loss'][df['dataset_id'] == i].mean()
        std_kld = df['kld_loss'][df['dataset_id'] == i].std()
        print(f"[13/18] Average latent space kld loss: {avg_kld}")
        print(f"[14/18] Std deviation of latent space kld loss: {std_kld}")
        avg_mse = df['mse_loss'][df['dataset_id'] == i].mean()
        std_mse = df['mse_loss'][df['dataset_id'] == i].std()
        print(f"[15/18] Average latent space mse loss: {avg_mse}")
        print(f"[16/18] Std deviation of latent space mse loss: {std_mse}")
        avg_class = df['class_loss'][df['dataset_id'] == i].mean()
        std_class = df['class_loss'][df['dataset_id'] == i].std()
        print(f"[17/18] Average latent space class loss: {avg_class}")
        print(f"[18/18] Std deviation of latent space class loss: {std_class}")

def main():
    # Argument parser
    parser = ArgumentParser(description="Train a VAE model")
    parser.add_argument('--config', type=str, default=SCRIPTS / "config/base.yml", help='Path to the configuration file')
    parser.add_argument('--model', type=str, default=None, help='Path to the model configuration file')
    parser.add_argument('--detached', type=bool, default=None, help='Wether to detach the classifier from the encoder')
    parser.add_argument('--kld', type=float, default=None, help='The weight of the KLD loss')
    parser.add_argument('--mse', type=float, default=None, help='The weight of the MSE loss')
    parser.add_argument('--class_weight', type=float, default=None, help='The weight of the classification loss')
    parser.add_argument('--output', type=str, default=None, help='Path to the model configuration file')
    parser.add_argument('--dataset', type=str, default=None, help='Pytorch dataset name')
    parser.add_argument('--val_split', type=float, default=None, help='Train-validation dataset split ratio')
    args = parser.parse_args()

    # Initialize parser
    p = Parser(config_file= args.config, args=args)

    config = p.config

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize VAE model
    vae = VAE.load_from_checkpoint("logs/vae_experiment/version_18/checkpoints/epoch=116-step=146250.ckpt").to(device)
    vae.eval()

    # Initialize TensorBoard logger
    logger = p.logger()

    augmentations = []

    augmentations.append(transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    augmentations.append(transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    augmentations.append(transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(0.5,0.5,0.5,0.5),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    augmentations.append(transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    augmentations.append(transforms.Compose([
        transforms.RandomPerspective(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    augmentations.append(transforms.Compose([
        transforms.RandomCrop(16, padding=4),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    augmentations.append(transforms.Compose([
        transforms.ElasticTransform(alpha=250.),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    augmentations.append(transforms.Compose([
        transforms.RandomInvert(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))



    df, latent_vectors = latent_space(augmentations, vae)

    plot_latent_space(df)
    stat_latent_space(latent_vectors,df)

if __name__ == "__main__":
    main()