import os
import json
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np


def plot_model_output_vs_label(outputs, labels, labels_flooded, filename):
    num_images = 4  # Number of images to display
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))  # Create a grid of subplots

    # outputs = outputs.cpu().detach()
    outputs = outputs.cpu()
    labels = labels.cpu()

    for i in range(num_images):
        # Display true labels
        ax = axes[0, i]
        ax.imshow(labels[i], cmap='gray', vmin=0, vmax=1)  # grayscale
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            # ax.set_ylabel('True Label', rotation=0, size='large', labelpad=40)
            ax.set_ylabel('True Label', rotation=90, size='large')
            ax.yaxis.set_label_position("left")
        if labels_flooded[i]:
            ax.set_title('Flood')
        else:
            ax.set_title('No Flood')

        # Display model outputs
        ax = axes[1, i]
        ax.imshow(outputs[i], cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            # ax.set_ylabel('Model Output', rotation=0, size='large', labelpad=40)
            ax.set_ylabel('Model Output', rotation=90, size='large')
            ax.yaxis.set_label_position("left")

    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_model_output_vs_label_square(outputs, labels, labels_flooded, filename):
    num_images = 8  # Number of images to display
    fig, axes = plt.subplots(int(num_images/2), int(num_images/2), figsize=(num_images, num_images))  # Create a grid of subplots

    if isinstance(outputs, list):
        outputs = [i.cpu() for i in outputs]
        labels = [j.cpu() for j in labels]
    else:
        outputs = outputs.cpu().detach()
        labels = labels.cpu()

    for i in range(num_images):
        row = (i//4)*2
        col = i % 4

        # Display true labels
        ax = axes[row, col]
        ax.imshow(labels[i], cmap='gray', vmin=0, vmax=1)  # grayscale
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            # ax.set_ylabel('True Label', rotation=0, size='large', labelpad=40)
            ax.set_ylabel('True Label', rotation=90, size='large')
            ax.yaxis.set_label_position("left")
        if labels_flooded[i]:
            ax.set_title('Flood')
        else:
            ax.set_title('No Flood')

        # Display model outputs
        ax = axes[row+1, col]
        ax.imshow(outputs[i], cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            # ax.set_ylabel('Model Output', rotation=0, size='large', labelpad=40)
            ax.set_ylabel('Model Output', rotation=90, size='large')
            ax.yaxis.set_label_position("left")

    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()



def plot_final_model_output_vs_label(model_names, outputs, labels, labels_flooded, filename):
    num_images = len(labels)  # Number of images to display
    # fig, axes = plt.subplots(len(model_names) + 1, num_images, figsize=(num_images*3, (len(model_names) + 1)*3))  # Create a grid of subplots; figsize is columns, rows
    fig = plt.figure(figsize=(num_images*2.5, (len(model_names) + 1)*2.5))  # Create a grid of subplots; figsize is columns, rows

    gs = GridSpec(len(model_names) + 1, num_images, figure=fig, wspace=0.000, hspace=0.02)

    outputs = [[i.cpu() for i in output] for output in outputs]
    labels = [l.cpu() for l in labels]

    for i in range(num_images):
        # Display true labels
        # ax = axes[0, i]
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(labels[i], cmap='gray', vmin=0, vmax=1)  # grayscale
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel('Ground Truth', rotation=90, size='large')
            # ax.set_ylabel('Ground Truth', rotation=90, fontsize=10)
            ax.yaxis.set_label_position("left")
        if labels_flooded[i]:
            ax.set_title('Flood')
        else:
            ax.set_title('No Flood')

        # Display model outputs
        for j in range(len(outputs)):
            # ax = axes[j+1, i]
            ax = fig.add_subplot(gs[j + 1, i])
            ax.imshow(outputs[j][i], cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                # ax.set_ylabel('Model Output', rotation=0, size='large', labelpad=40)
                ax.set_ylabel(f"{model_names[j]} Output", rotation=90, size='large')
                # ax.set_ylabel(f"{model_names[j]} Output", rotation=90, fontsize=10)
                ax.yaxis.set_label_position("left")

    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.001)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_loss_chart(losses, epochs, filename, hyperparams):
    plt.figure(figsize=(10, 6))
    labels = ['train', 'validation', 'test']
    for i, series in enumerate(losses):
        plt.plot(epochs, series, marker=None, linestyle='-', label=f'{labels[i]}')
   
    # Customize the plot
    plt.title("Model Loss Chart")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplots_adjust(right=0.75)
    plt.ylim(0, 1)

    hyperparams_text = '\n'.join([f'{key}: {value}' for key, value in hyperparams.items()])
    resolution_match = re.search(r'res(\d+)', filename)
    resolution = int(resolution_match.group(1))
    hyperparams_text = hyperparams_text + f"\nResolution: {resolution}"
    # plt.figtext(0.15, -0.2, "Hyperparameters:\n" + hyperparams_text, fontsize=9, 
    #             bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))
    plt.figtext(0.76, 0.5, "Hyperparameters:\n" + hyperparams_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'), 
                ha='left', va='center')
    
    plt.legend(loc='upper right')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def strip_black_pixel_padding_PIL(config_file_path: str, resolution: int, image_path: str):
    with open(config_file_path) as core_config_file:
        core_config = json.load(core_config_file)
    
    dimension_string = core_config[f"rainfall_reprojection_master_{resolution}"]
    match = re.search(r'_(\d+)_(\d+)\.tif$', dimension_string)
    new_dimension_right, new_dimension_bottom = int(match.group(1)), int(match.group(2))

    image = Image.open(image_path)
    crop_area = (0, 0, new_dimension_right, new_dimension_bottom)
    cropped_image = image.crop(crop_area)

    #convert to numpy
    cropped_image = np.array(cropped_image)

    return cropped_image