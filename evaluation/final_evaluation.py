from evaluation.model_evaluation_helpers import *
from model_runs.model_run_helpers import *
from model_runs.distributed_gpu_helpers import *
from visualisations.visualisation_helpers import *

if __name__=="__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    torch.manual_seed(42)
    resolution = 256
    # Set up config files
    with open(os.environ["PROJECT_FLOOD_CORE_PATHS"]) as core_config_file:
        core_config = json.load(core_config_file)
    with open(os.environ["PROJECT_FLOOD_DATA"]) as data_config_file:
        data_config = json.load(data_config_file)

    # Load newly trained best, final models
    final_sep_branch_model, _, _, final_sep_branch_params = load_checkpoint(os.path.join(data_config['saved_models_path'], 'ConvLSTMSeparateBranches_epochs2000_batchsize8_lr0p001_precedingrainfall3_dropout0p25_outputchannels16_convblocklayers2_convLSTMlayers1_optimRMSprop_criterionBCELoss_transformsFalse_res256_20240830_1046_earlystop.pt')) #TODO FILL IN FILEPATHS
    final_merged_model, _, _, final_merged_params = load_checkpoint(os.path.join(data_config['saved_models_path'], 'ConvLSTMMerged_epochs2000_batchsize8_lr0p001_precedingrainfall1_dropout0p25_outputchannels16_convblocklayers2_convLSTMlayers2_optimRMSprop_criterionBCELoss_transformsFalse_res256_20240901_1066_earlystop.pt'))

    # Set up test_dataloaders
    sep_branch_test_dataloader = get_dataloader("test_labels_path", resolution=256, preceding_rainfall_days=final_sep_branch_params['precedingrainfall'], forecast_rainfall_days=1, 
                                                transform=None, batch_size=16, shuffle=False, num_workers=4)
    
    merged_test_dataloader = get_dataloader("test_labels_path", resolution=256, preceding_rainfall_days=final_merged_params['precedingrainfall'], forecast_rainfall_days=1, 
                                                transform=None, batch_size=16, shuffle=False, num_workers=4)
    
    # Get cropping dimensions
    dimension_string = core_config[f"rainfall_reprojection_master_{resolution}"]
    match = re.search(r'_(\d+)_(\d+)\.tif$', dimension_string)
    new_dimension_right, new_dimension_bottom = int(match.group(1)), int(match.group(2))

    # Get mask
    mask_path = os.path.join(data_config["model_results_path"], f"perm_water_mask_{resolution}.npy")
    perm_water_mask = np.load(os.path.join(data_config["model_results_path"], f"perm_water_mask_{resolution}.npy"))

    # Calculate loss metrics, one epoch
    print("========= SEPARATE BRANCH ==============")
    sep_branch_metrics = evaluate_model(final_sep_branch_model, sep_branch_test_dataloader, final_sep_branch_params['criterion'], device, mask_path, new_dimension_right, new_dimension_bottom)
    print("\n\n")
    print("========= MERGED BRANCH ==============")
    merged_metrics = evaluate_model(final_merged_model, merged_test_dataloader, final_sep_branch_params['criterion'], device, mask_path, new_dimension_right, new_dimension_bottom)
    print("\n\n")

    # Plot loss metrics by threshold
    metric_combo = [sep_branch_metrics, merged_metrics]
    titles = ['Branched Model', 'Merged Model']
    threshold_metrics_filename = os.path.join(data_config["model_results_path"], f"metrics_by_threshold_{resolution}.png")
    plot_metrics_vs_thresholds(metric_combo, threshold_metrics_filename, titles)

    #Plot ROC
    roc_filename = os.path.join(data_config["model_results_path"], f"ROC_curves_{resolution}.png")
    plot_roc_auc_curves(metric_combo, roc_filename, titles)

    # Save image wide metrics
    metrics_filename = os.path.join(data_config["model_results_path"], f"key_metrics_{resolution}.csv")
    save_metrics_to_csv(metric_combo, metrics_filename, titles)

    # Plot test images for each model
    with torch.no_grad():
        size = 3
        sep_branch_flooded_images, sep_branch_non_flooded_images = collect_images(final_sep_branch_model, sep_branch_test_dataloader, size)
        merged_flooded_images, merged_non_flooded_images = collect_images(final_merged_model, merged_test_dataloader, size)

        selected_sep_branch_outputs = [img[0] for img in sep_branch_flooded_images + sep_branch_non_flooded_images]
        selected_sep_branch_outputs = [i[:new_dimension_bottom, :new_dimension_right] for i in selected_sep_branch_outputs] #crop

        selected_merged_outputs = [img[0] for img in merged_flooded_images + merged_non_flooded_images]
        selected_merged_outputs = [i[:new_dimension_bottom, :new_dimension_right] for i in selected_merged_outputs] #crop

        model_names = ['Branched', 'Merged']
        selected_model_outputs = [selected_sep_branch_outputs, selected_merged_outputs]

        selected_targets = [img[1] for img in sep_branch_flooded_images + sep_branch_non_flooded_images]
        selected_targets = [i[:new_dimension_bottom, :new_dimension_right] for i in selected_targets] #crop

        selected_targets_flooded = [img[2] for img in sep_branch_flooded_images + sep_branch_non_flooded_images] #boolean for flooded or not

        # Do plotting
        plot_filename = os.path.join(data_config["model_results_path"], "final_plots.png")
        plot_final_model_output_vs_label(model_names, selected_model_outputs, selected_targets, selected_targets_flooded, plot_filename)

        # Risk chart
        crs_transform = [0.0226492347869873, 0.0, 88.08430518433968, 0.0, -0.0226492347869873, 26.44864775268901]
        selected_targets_risk = [np.ma.masked_array(i, mask=perm_water_mask) for i in selected_targets]
        selected_sep_branch_outputs_risk = [np.ma.masked_array(i, mask=perm_water_mask) for i in selected_sep_branch_outputs]
        selected_merged_outputs_risk = [np.ma.masked_array(i, mask=perm_water_mask) for i in selected_merged_outputs]
        selected_model_outputs_risk = [selected_sep_branch_outputs_risk, selected_merged_outputs_risk]

        plot_filename = os.path.join(data_config["model_results_path"], "final_plots_risk.png")
        plot_risk_on_map(model_names, selected_model_outputs_risk, selected_targets_risk, selected_targets_flooded, plot_filename, crs_transform)

        # Pixel differences
        for i in range(len(model_names)):
            plot_filename = os.path.join(data_config["model_results_path"], f"final_plots_{model_names[i]}_pixel_difference.png")
            plot_pixel_difference(model_names[i], selected_model_outputs[i], selected_targets, selected_targets_flooded, plot_filename)
