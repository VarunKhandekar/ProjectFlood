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
        ax.imshow(labels[i], cmap='gray')  # Assuming grayscale images; change cmap if using color images
        ax.axis('off')  # Hide axes ticks
        ax.set_title('True Label')

        # Display model outputs
        ax = axes[1, i]
        ax.imshow(outputs[i], cmap='gray')
        ax.axis('off')
        ax.set_title('Model Output')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
