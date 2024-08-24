from final_evaluation.model_evaluation_helpers import *
from model_runs.model_run_helpers import *
from model_runs.distributed_gpu_helpers import *
from visualisations.visualisation_helpers import *

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    # Load newly trained best, final models
    final_sep_branch_model, _, _, final_sep_branch_params = load_checkpoint("...") #TODO FILL IN FILEPATHS
    final_merged_model, _, _, final_merged_params = load_checkpoint("...")


    # with open(os.environ["PROJECT_FLOOD_DATA"]) as data_config_file:
    #     data_config = json.load(data_config_file)
        
    # final_sep_branch_fp = os.path.join(data_config["saved_models_path"], f"{get_attribute(sep_branch_model, 'name')}_{best_sep_branch_params['epochs']}_FINAL.pt")
    # final_merged_fp = os.path.join(data_config["saved_models_path"], f"{get_attribute(merged_model, 'name')}_{best_merged_params['epochs']}_FINAL.pt")

    # final_sep_branch_model, _, _, final_sep_branch_params = load_checkpoint(final_sep_branch_fp)
    # final_merged_branch_model_, _, _, final_merged_params = load_checkpoint(final_merged_fp)

    # Set up test_dataloaders
    sep_branch_test_dataloader = get_dataloader("test_labels_path", resolution=256, preceding_rainfall_days=final_sep_branch_params['precedingrainfall'], forecast_rainfall_days=1, 
                                                transform=None, batch_size=16, shuffle=False, num_workers=4)
    
    merged_test_dataloader = get_dataloader("test_labels_path", resolution=256, preceding_rainfall_days=final_merged_params['precedingrainfall'], forecast_rainfall_days=1, 
                                                transform=None, batch_size=16, shuffle=False, num_workers=4)
    

    # Calculate loss metrics, one epoch
    evaluate_model(os.environ["PROJECT_FLOOD_DATA"], final_sep_branch_model, sep_branch_test_dataloader, final_sep_branch_params['criterion'], device)
    evaluate_model(os.environ["PROJECT_FLOOD_DATA"], final_merged_model, merged_test_dataloader, final_sep_branch_params['criterion'], device)


    # Plot image
    with torch.no_grad():
        size = 3

        sep_branch_flooded_images = []
        sep_branch_non_flooded_images = []
        for inputs, targets, flooded in sep_branch_test_dataloader:
            sep_branch_outputs = final_sep_branch_model(inputs)
            for i in range(len(flooded)):
                if flooded[i] == 1 and len(sep_branch_flooded_images) < size:
                    sep_branch_flooded_images.append((sep_branch_outputs[i], targets[i], flooded[i]))
                elif flooded[i] == 0 and len(sep_branch_non_flooded_images) < size:
                    sep_branch_non_flooded_images.append((sep_branch_outputs[i], targets[i], flooded[i]))
                # Stop if we've collected 4 images in each category
                if len(sep_branch_flooded_images) >= size and len(sep_branch_non_flooded_images) >= size:
                    break
            if len(sep_branch_flooded_images) >= 4 and len(sep_branch_non_flooded_images) >= 4:
                break

        merged_flooded_images = []
        merged_non_flooded_images = []
        for inputs, targets, flooded in merged_test_dataloader:
            merged_outputs = final_merged_model(inputs)
            for i in range(len(flooded)):
                if flooded[i] == 1 and len(merged_flooded_images) < size:
                    merged_flooded_images.append((merged_outputs[i], targets[i], flooded[i]))
                elif flooded[i] == 0 and len(merged_non_flooded_images) < size:
                    merged_non_flooded_images.append((merged_outputs[i], targets[i], flooded[i]))
                # Stop if we've collected 4 images in each category
                if len(merged_flooded_images) >= size and len(merged_non_flooded_images) >= size:
                    break
            if len(merged_flooded_images) >= 4 and len(merged_non_flooded_images) >= 4:
                break

        selected_sep_branch_outputs = [img[0] for img in sep_branch_flooded_images + sep_branch_non_flooded_images]
        selected_merged_outputs = [img[0] for img in merged_flooded_images + merged_non_flooded_images]

        model_names = ['Branched', 'Merged']
        selected_model_outputs = [selected_sep_branch_outputs, selected_merged_outputs]
        selected_targets = [img[1] for img in sep_branch_flooded_images + sep_branch_non_flooded_images]
        selected_targets_flooded = [img[2] for img in sep_branch_flooded_images + sep_branch_non_flooded_images]

        # Do plotting
        with open(os.environ["PROJECT_FLOOD_DATA"]) as data_config_file:
            data_config = json.load(data_config_file)
        plot_filename = os.path.join(data_config["model_results_path"], "final_plots.png")
        plot_final_model_output_vs_label(model_names, selected_model_outputs, selected_targets, selected_targets_flooded, plot_filename)
