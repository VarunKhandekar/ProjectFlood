import os
import json
import matplotlib.pyplot as plt


def plot_model_output_vs_label(outputs, labels, filename):
    num_images = 4  # Number of images to display
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))  # Create a grid of subplots

    outputs = outputs.cpu().detach()
    labels = labels.cpu()

    for i in range(num_images):
        # Display true labels
        ax = axes[0, i]
        ax.imshow(labels[i], cmap='gray', vmin=0, vmax=1)  # Assuming grayscale images; change cmap if using color images
        ax.axis('off')  # Hide axes ticks
        if i == 0:
            ax.set_ylabel('True Label', rotation=0, size='large', labelpad=40)
            ax.yaxis.set_label_position("left")

        # Display model outputs
        ax = axes[1, i]
        ax.imshow(outputs[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Model Output', rotation=0, size='large', labelpad=40)
            ax.yaxis.set_label_position("left")

    plt.tight_layout()
    plt.savefig(filename)
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

    hyperparams_text = '\n'.join([f'{key}: {value}' for key, value in hyperparams.items()])
    plt.figtext(0.15, 0.2, "Hyperparameters:\n" + hyperparams_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))
    
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()