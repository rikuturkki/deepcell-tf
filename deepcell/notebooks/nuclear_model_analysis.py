import os
import papermill as pm

filename = 'general_nuclear.npz'
date = '04262020'

dataset_split_seeds = [0] # [0, 1, 2]
dataset_fractions = [0.5] #[0.01, 0.1, 0.25, 0.5, 1]
backbones = ['resnet50'] #'resnet50', 'mobilenetv2']
model_types = ['pixelwise', 'watershed'] #'retinamask', 'pixelwise', 'watershed']

for seed in dataset_split_seeds:
    for dataset_fraction in dataset_fractions:
        for backbone in backbones:
            for model_type in model_types:
                filename = 'nuclear_{}_{}_{}_{}'.format(seed, dataset_fraction, backbone, model_type)
                
                print('Processing {}'.format(filename))
                
                # Define notebook paths
                input_notebook_name = 'Papermill - Nuclear Accuracy - Proper Split.ipynb'
                output_notebook_name = filename + '_training_notebook.ipynb'
                
                input_direc = os.path.join('/notebooks', 'papermill')
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
                              'train_permutation_seed': seed,
                              'dataset_fraction': dataset_fraction,
                              'backbone':backbone,
                              'model_type': model_type}
                
                # Run notebook
                pm.execute_notebook(input_path,
                                    output_path,
                                    parameters=parameters)
