import os
import json
import re
import matplotlib.pyplot as plt


def plot_model_output_vs_label(outputs, labels, labels_flooded, filename):
    num_images = 4  # Number of images to display
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))  # Create a grid of subplots

    outputs = outputs.cpu().detach()
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