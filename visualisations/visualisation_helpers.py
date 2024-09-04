import json
import re
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import auc
from PIL import Image
import numpy as np


def plot_model_output_vs_label(outputs: torch.Tensor, labels: torch.Tensor, labels_flooded: list, filename: str) -> None:
    """
    Plot and compare model outputs with true labels, indicating whether the region was flooded or not, and save the plot as an image file.

    Args:
        outputs (torch.Tensor): The model's predicted outputs.
        labels (torch.Tensor): The ground truth labels.
        labels_flooded (list): A list indicating whether each sample corresponds to a flooded region (True/False).
        filename (str): Path and name of the file where the plot will be saved.

    Returns:
        None: The function generates and saves a grid plot comparing model outputs with true labels.
    
    """
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


def plot_model_output_vs_label_square(outputs: torch.Tensor, labels: torch.Tensor, labels_flooded: list, filename: str) -> None:
    """
    Plot and compare model outputs with true labels in a square grid format, indicating whether the region was flooded or not, and save the plot as an image file.

    Args:
        outputs (torch.Tensor or list): The model's predicted outputs, either as a tensor or a list of tensors.
        labels (torch.Tensor or list): The ground truth labels, either as a tensor or a list of tensors.
        labels_flooded (list): A list indicating whether each sample corresponds to a flooded region (True/False).
        filename (str): Path and name of the file where the plot will be saved.

    Returns:
        None: The function generates and saves a grid plot comparing model outputs with true labels.

    """
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


def plot_loss_chart(losses: list, epochs: list, filename: str, hyperparams: dict) -> None:
    """
    Plot the training and validation loss over epochs and save the chart as an image file.

    Args:
        losses (list): A list of loss values for each phase (e.g., training, validation) across epochs.
        epochs (list): A list of epoch numbers corresponding to the loss values.
        filename (str): The path and name of the file where the plot will be saved.
        hyperparams (dict): A dictionary containing hyperparameters to display on the plot.

    Returns:
        None: The function generates and saves a line plot of the loss over epochs, including hyperparameters as text on the plot.
    
    """
    plt.figure(figsize=(10, 6))
    labels = ['train', 'validation']
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


def strip_black_pixel_padding_PIL(config_file_path: str, resolution: int, image_path: str) -> np.ndarray:
    """
    Remove black pixel padding from an image based on the specified resolution and return the cropped image as a NumPy array.

    Args:
        config_file_path (str): Path to the configuration file containing image dimension information.
        resolution (int): The resolution of the image for which padding should be removed.
        image_path (str): Path to the image file that needs to be cropped.

    Returns:
        np.ndarray: The cropped image as a NumPy array without black pixel padding.

    """
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


def plot_pixel_difference(model_name: str, outputs: list, labels: list, labels_flooded: list, filename: str, threshold: float = 0.5) -> None:
    """
    Plot and compare model predictions with true labels, showing pixel-level differences, and save the plot as an image file.

    Args:
        model_name (str): The name of the model being evaluated.
        outputs (list): A list of model output tensors or arrays.
        labels (list): A list of ground truth label tensors or arrays.
        labels_flooded (list): A list indicating whether each sample corresponds to a flooded region (True/False).
        filename (str): The path and name of the file where the plot will be saved.
        threshold (float, optional): The threshold value for binary classification of outputs. Default is 0.5.

    Returns:
        None: The function generates and saves a grid plot showing the true labels, model predictions, and pixel differences.
    
    """
    if isinstance(outputs[0], torch.Tensor):
        outputs = [i.cpu().numpy() for i in outputs] 
    if isinstance(labels[0], torch.Tensor):
        labels = [l.cpu().numpy() for l in labels]

    difference_cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"], gamma=2)
    # difference_cmap = mcolors.LinearSegmentedColormap.from_list("",  [
    #     (0.0, "white"),  # 0% at white
    #     (0.3**2, "white"),  # ~9% still white (adjust the exponent for the effect)
    #     (0.6**2, "pink"),  # ~36% pinkish
    #     (1.0, "red"),    # 100% at red
    # ])

    # Create a grid of subplots; figsize is columns, rows
    x, y = labels[0].shape
    width_adjustment = (y/x)*1.03
    num_images = len(labels)
    scale_factor = 2
    fig, axes = plt.subplots(num_images, 3, figsize=(3*width_adjustment*scale_factor, num_images*scale_factor)) 

    for i in range(num_images):
        # Truth
        ax = axes[i, 0]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(labels[i], cmap='gray', vmin=0, vmax=1)
        if i == 0:
            ax.set_title('Truth')
        if labels_flooded[i]:
            ax.set_ylabel('Flood', rotation=90, size='large')
            ax.yaxis.set_label_position("left")
        else:
            ax.set_ylabel('No Flood', rotation=90, size='large')
            ax.yaxis.set_label_position("left")
        
        # Output
        ax = axes[i, 1]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(outputs[i], cmap='gray', vmin=0, vmax=1)
        if i == 0:
            ax.set_title('Prediction')
        # Pixel difference
        ax = axes[i, 2]
        ax.set_xticks([])
        ax.set_yticks([])
        # thresholded_output = outputs[i] > threshold
        absolute_pixel_difference = np.abs(outputs[i] - labels[i])
        # absolute_pixel_difference = np.abs(thresholded_output - labels[i])
        ax.imshow(absolute_pixel_difference, cmap=difference_cmap, vmin=0, vmax=1)
        if i == 0:
            ax.set_title('Difference')

    fig.suptitle(f"Pixel Difference: {model_name}", fontsize=16, y=0.93)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_final_model_output_vs_label(model_names: list, outputs: list, labels: list, labels_flooded: list, filename: str, risk: bool = False) -> None:
    """
    Plot and compare model outputs from multiple models with true labels, displaying whether regions were flooded or not, and save the plot as an image file.

    Args:
        model_names (list): A list of model names corresponding to each set of outputs.
        outputs (list): A list of lists, where each inner list contains the outputs from a model.
        labels (list): A list of ground truth label tensors or arrays.
        labels_flooded (list): A list indicating whether each sample corresponds to a flooded region (True/False).
        filename (str): The path and name of the file where the plot will be saved.
        risk (bool, optional): If True, displays outputs with a risk colormap. Default is False.

    Returns:
        None: The function generates and saves a grid plot comparing model outputs with true labels.

    """
    num_images = len(labels)  # Number of images to display
    # fig, axes = plt.subplots(len(model_names) + 1, num_images, figsize=(num_images*3, (len(model_names) + 1)*3))  # Create a grid of subplots; figsize is columns, rows
    fig = plt.figure(figsize=(num_images*2.5, (len(model_names) + 1)*2.5))  # Create a grid of subplots; figsize is columns, rows
    gs = GridSpec(len(model_names) + 1, num_images, figure=fig, wspace=0.000, hspace=0.02)

    if isinstance(outputs[0][0], torch.Tensor):
        outputs = [[i.cpu().numpy() for i in output] for output in outputs]
    if isinstance(labels[0], torch.Tensor):
        labels = [l.cpu().numpy() for l in labels]

    if risk:
        risk_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'green'), (0.5, 'yellow'), (1, 'red')])
        risk_cmap.set_bad(color='black')

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
            if risk:
                im = ax.imshow(outputs[j][i], cmap=risk_cmap, vmin=0, vmax=1)
            else:
                im = ax.imshow(outputs[j][i], cmap='gray', vmin=0, vmax=1)  # grayscale
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                # ax.set_ylabel('Model Output', rotation=0, size='large', labelpad=40)
                ax.set_ylabel(f"{model_names[j]} Output", rotation=90, size='large')
                # ax.set_ylabel(f"{model_names[j]} Output", rotation=90, fontsize=10)
                ax.yaxis.set_label_position("left")

    if risk:
        cbar = fig.colorbar(im, ax=fig.get_axes(), orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Probability of Flooding', rotation=270, labelpad=15)
    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.001)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_risk_on_map(model_names: list, outputs: list, labels: list, labels_flooded: list, filename: str, crs_transform: list) -> None:
    """
    Plot and compare model outputs from multiple models with true labels on a map, showing flood risk, and save the plot as an image file.

    Risk maps are overlaid on top of the appropriate world location using the crs_transform parameters provided.

    Args:
        model_names (list): A list of model names corresponding to each set of outputs.
        outputs (list): A list of lists, where each inner list contains the outputs from a model.
        labels (list): A list of ground truth label tensors or arrays.
        labels_flooded (list): A list indicating whether each sample corresponds to a flooded region (True/False).
        filename (str): The path and name of the file where the plot will be saved.
        crs_transform (list): A list of transformation parameters for mapping coordinates.

    Returns:
        None: The function generates and saves a map-based grid plot comparing model outputs with true labels.

    """
    num_images = len(labels)  # Number of images to display
    # fig, axes = plt.subplots(len(model_names) + 1, num_images, figsize=(num_images*3, (len(model_names) + 1)*3))  # Create a grid of subplots; figsize is columns, rows
    fig = plt.figure(figsize=(num_images*4, (len(model_names) + 1)*4))  # Create a grid of subplots; figsize is columns, rows
    gs = GridSpec(len(model_names) + 1, num_images, figure=fig, wspace=0.002, hspace=0.0002)

    if isinstance(outputs[0][0], torch.Tensor):
        outputs = [[i.cpu().numpy() for i in output] for output in outputs]
    if isinstance(labels[0], torch.Tensor):
        labels = [l.cpu().numpy() for l in labels]

    risk_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'green'), (0.5, 'yellow'), (1, 'red')])
    risk_cmap.set_bad(color='black')

    extent = [
        crs_transform[2],  # left (min longitude)
        crs_transform[2] + crs_transform[0] * labels[0].shape[1],  # right (max longitude)
        crs_transform[5] + crs_transform[4] * labels[0].shape[0],  # bottom (min latitude)
        crs_transform[5]  # top (max latitude)
    ]

    for i in range(num_images):
        # Display true labels
        # Set up subplot
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())  
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Set gridlines
        gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', 
                          alpha=0.6, linestyle='--')
        gl.top_labels = False  
        gl.right_labels = False 
        gl.bottom_labels = False
        gl.left_labels = False
        if i == 0:
            gl.left_labels = True
        gl.ylabel_style = {'size': 10, 'color': 'black'}
        gl.xlocator = plt.MaxNLocator(4)
        gl.ylocator = plt.MaxNLocator(5)

        im = ax.imshow(labels[i], extent=extent, transform=ccrs.PlateCarree(), cmap=risk_cmap, vmin=0, vmax=1, 
                       alpha=0.75, interpolation='none')

        if i == 0:
            ax.text(-0.25, 0.5, 'Ground Truth', va='center', ha='right', 
                    fontsize=12, transform=ax.transAxes, rotation=90)
        if labels_flooded[i]:
            ax.set_title('Flood')
        else:
            ax.set_title('No Flood')

        # Display model outputs
        for j in range(len(outputs)):
            # Set up subplot
            ax = fig.add_subplot(gs[j + 1, i], projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            #Set gridlines
            gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', 
                              alpha=0.6, linestyle='--')
            gl.bottom_labels = False
            gl.left_labels = False 
            gl.top_labels = False  
            gl.right_labels = False 
            if j == len(outputs)-1:
                gl.bottom_labels = True
            if i == 0:
                gl.left_labels = True
            gl.xlabel_style = {'size': 10, 'color': 'black'}
            gl.ylabel_style = {'size': 10, 'color': 'black'}
            gl.xlocator = plt.MaxNLocator(4)  # Reduce number of longitude labels
            gl.ylocator = plt.MaxNLocator(5)

            im = ax.imshow(outputs[j][i], extent=extent, transform=ccrs.PlateCarree(), cmap=risk_cmap, vmin=0, vmax=1, 
                           alpha=0.75, interpolation='none')

            if i == 0:
                ax.text(-0.25, 0.5, f"{model_names[j]} Output", va='center', ha='right', 
                    fontsize=12, transform=ax.transAxes, rotation=90)


    cbar = fig.colorbar(im, ax=fig.get_axes(), orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Probability of Flooding', rotation=270, labelpad=15)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_metrics_vs_thresholds(metric_accumulators: list, filename: str, titles: list = None) -> None:
    """
    Plot Precision, Recall, Accuracy, and F1 Scores against thresholds for various models and save the plot as an image file.

    The relevant data points for the models are saved in the dictionaries in the metric_accumulators list.

    Args:
        metric_accumulators (list): A list of dictionaries, where each dictionary contains performance metrics ('precision_scores', 'recall_scores', 'accuracy_scores', 'f1_scores') for different thresholds.
        filename (str): The path and name of the file where the plot will be saved.
        titles (list, optional): A list of titles for each subplot. If not provided, subplots will be numbered automatically.

    Returns:
        None: The function generates and saves a plot showing how metrics change across thresholds.

    """
    
    num_plots = len(metric_accumulators)
    
    # Create subplots
    fig, axs = plt.subplots(1, num_plots, figsize=(15, 6), sharey=True)
    
    # If only one plot, convert axs to a list to maintain consistency
    if num_plots == 1:
        axs = [axs]

    for i, metric_accumulator in enumerate(metric_accumulators):
        # Extract thresholds, precision, recall, accuracy, and F1 scores from the metric_accumulator
        thresholds = list(metric_accumulator['precision_scores'].keys())
        precision_scores = list(metric_accumulator['precision_scores'].values())
        recall_scores = list(metric_accumulator['recall_scores'].values())
        accuracy_scores = list(metric_accumulator['accuracy_scores'].values())
        f1_scores = list(metric_accumulator['f1_scores'].values())

        axs[i].plot(thresholds, precision_scores, label='Precision')
        axs[i].plot(thresholds, recall_scores, label='Recall')
        axs[i].plot(thresholds, accuracy_scores, label='Accuracy')
        axs[i].plot(thresholds, f1_scores, label='F1 Score')

        if titles:
            axs[i].set_title(titles[i])
        axs[i].set_xlabel('Threshold')
        if i == 0:  # Only set ylabel for the first subplot for clarity
            axs[i].set_ylabel('Score')
        
        axs[i].legend()
        axs[i].grid(True)

    fig.suptitle("Performance Metrics by Threshold", fontsize=15, y=0.97)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_roc_auc_curves(metric_accumulators: list, filename: str, titles: list = None) -> None:
    """
    Plot the ROC (Receiver Operating Characteristic) curve and calculate the AUC (Area Under the Curve) for various models, saving the plot as an image file.

    The relevant data points for the models are saved in the dictionaries in the metric_accumulators list.

    Args:
        metric_accumulators (list): A list of dictionaries, where each dictionary contains 'false_positive_rates' and 'recall_scores' (True Positive Rate) for different thresholds.
        filename (str): The path and name of the file where the plot will be saved.
        titles (list, optional): A list of titles for each subplot. If not provided, subplots will be numbered automatically.

    Returns:
        None: The function generates and saves a plot showing the ROC curve and AUC for each metric accumulator.

    """    
    num_plots = len(metric_accumulators)

    fig, axs = plt.subplots(1, num_plots, figsize=(15, 6), sharey=True)
    
    # If only one plot, convert axs to a list to maintain consistency
    if num_plots == 1:
        axs = [axs]

    for i, metric_accumulator in enumerate(metric_accumulators):
        fpr = list(metric_accumulator['false_positive_rates'].values())
        tpr = list(metric_accumulator['recall_scores'].values())
        
        roc_auc = auc(fpr, tpr) # Calculate AUC
        
        axs[i].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axs[i].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)') # random classifier
        
        if titles:
            axs[i].set_title(titles[i])
        axs[i].set_xlabel('False Positive Rate (FPR)')
        if i == 0:
            axs[i].set_ylabel('True Positive Rate (TPR)')
        
        axs[i].legend(loc='lower right')
        axs[i].grid(True)

    fig.suptitle("ROC Curves", fontsize=15, y=0.97)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
