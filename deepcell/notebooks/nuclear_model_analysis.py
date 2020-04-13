import os
import papermill as pm

filename = 'general_nuclear_train.npz'
date = '04122020'

dataset_split_seeds = [0] # [0, 1, 2]
dataset_sizes = [128] #[128, 512, 2048, 8192, 32768, 82800]
backbones = ['resnet50'] # ['resnet50', 'mobilenetv2']
model_types = ['retinamask', 'pixelwise', 'watershed']

for seed in dataset_split_seeds:
    for dataset_size in dataset_sizes:
        for backbone in backbones:
            for model_type in model_types:
                filename = 'nuclear_{}_{}_{}_{}'.format(seed, dataset_size, backbone, model_type)
                
                print('Processing {}'.format(filename))
                
                # Define notebook paths
                input_notebook_name = 'Papermill - Nuclear Accuracy.ipynb'
                output_notebook_name = filename + '_training_notebook.ipynb'
                
                input_direc = os.path.join('/notebooks','papermill')
                output_direc = os.path.join('/notebooks', date)
                
                os.makedirs(input_direc, exist_ok=True)
                os.makedirs(output_direc, exist_ok=True)
                
                input_path = os.path.join(input_direc, input_notebook_name)
                output_path = os.path.join(output_direc, output_notebook_name)
                
                # Define parameters
                parameters = {'n_epochs':16, 
                              'batch_size':4 if model_type == 'retinamask' else 16, 
                              'date': date, 
                              'filename': filename,
                              'dataset_split_seed': seed,
                              'dataset_size': dataset_size,
                              'backbone':backbone,
                              'model_type': model_type}
                
                # Run notebook
                pm.execute_notebook(input_path,
                                    output_path,
                                    parameters=parameters)
