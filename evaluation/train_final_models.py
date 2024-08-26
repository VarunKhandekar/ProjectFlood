import datetime
from evaluation.model_evaluation_helpers import *
from model_runs.model_run_helpers import *
from model_runs.distributed_gpu_helpers import *


if __name__=="__main__":
    model_run_date = datetime.date.today().strftime(r'%Y%m%d')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    torch.manual_seed(42)

    #     hyperparams = {
    #     'epochs': num_epochs,
    #     'batchsize': train_dataloader.batch_size,
    #     'lr': lr,
    #     'precedingrainfall': model.preceding_rainfall_days,
    #     'dropout': model.dropout_prob,
    #     'outputchannels': model.output_channels,
    #     'convblocklayers': model.conv_block_layers,
    #     'convLSTMlayers': model.convLSTM_layers,
    #     'optim': optimizer_type,
    #     'criterion': criterion_type,
    #     'transforms': True if train_dataloader.dataset.transform else False,
    #     'res': train_dataloader.dataset.resolution
    # }

    # Load hyperparams for best models
    _, _, _, best_sep_branch_params = load_checkpoint("...") #TODO ADD IN FILEPATHS
    _, _, _, best_merged_params = load_checkpoint("...")


    # Set up models
    sep_branch_model = ConvLSTMSeparateBranches(best_sep_branch_params['precedingrainfall'], 1, best_sep_branch_params['outputchannels'], 
                                                best_sep_branch_params['convblocklayers'],  best_sep_branch_params['convLSTMlayers'], best_sep_branch_params['dropout'])
    sep_branch_model_name = generate_model_name(sep_branch_model.__class__.__name__, model_run_date, **best_sep_branch_params)
    sep_branch_model.name = f"{sep_branch_model_name}_BEST"


    merged_model = ConvLSTMMerged(best_merged_params['precedingrainfall'], 1, best_merged_params['outputchannels'], 
                                  best_merged_params['convblocklayers'], best_merged_params['convLSTMlayers'], best_merged_params['dropout'])
    merged_model_name = generate_model_name(merged_model.__class__.__name__, model_run_date, **best_merged_params)
    merged_model.name = f"{merged_model_name}_BEST"


    # Set up train+val datasets
    sep_branch_transform = train_transform if best_sep_branch_params['transforms'] else None
    train_dataset_sep_branch = FloodPredictionDataset(os.environ["PROJECT_FLOOD_DATA"], os.environ["PROJECT_FLOOD_CORE_PATHS"], 
                                            "training_validation_combo_path", resolution=256, preceding_rainfall_days=best_sep_branch_params['precedingrainfall'], 
                                                forecast_rainfall_days=1, transform=sep_branch_transform)


    merged_transform = train_transform if best_merged_params['transforms'] else None
    train_dataset_merged = FloodPredictionDataset(os.environ["PROJECT_FLOOD_DATA"], os.environ["PROJECT_FLOOD_CORE_PATHS"], 
                                            "training_validation_combo_path", resolution=256, preceding_rainfall_days=best_merged_params['precedingrainfall'], 
                                                forecast_rainfall_days=1, transform=merged_transform)

        
    # Train merged and separate_branches
    epochs = 2000 #Manually set as we have early stopping in place. What was previous optimal epochs might be different

    torch.multiprocessing.spawn(train_model_dist, args=(world_size, os.environ['PROJECT_FLOOD_DATA'], 
                                                        sep_branch_model, 
                                                        best_sep_branch_params['criterion'], 
                                                        best_sep_branch_params['optim'], 
                                                        best_sep_branch_params['lr'], 
                                                        best_sep_branch_params['epochs'],
                                                        False,
                                                        False,
                                                        best_sep_branch_params['batchsize'],
                                                        train_dataset_sep_branch,
                                                        True), nprocs=world_size)

    # Train the model
    torch.multiprocessing.spawn(train_model_dist, args=(world_size, os.environ['PROJECT_FLOOD_DATA'], 
                                                        merged_model, 
                                                        best_merged_params['criterion'], 
                                                        best_merged_params['optim'], 
                                                        best_merged_params['lr'], 
                                                        best_merged_params['epochs'],
                                                        False,
                                                        False,
                                                        best_merged_params['batchsize'],
                                                        train_dataset_merged,
                                                        True), nprocs=world_size)

