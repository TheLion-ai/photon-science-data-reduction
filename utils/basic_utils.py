import os

def prepare_path(model_name, dataset_name):
    """Create a path to store experiment results. If a folder in the path doesn't exist, it is created.

    Args:
        model_name (str): the name of the model that was used for the experiment.
        dataset_name ([type]): the name of the model that was used for the experiment.

    Returns:
        str: An existing path to experiment results for a given model and a dataset.
    """
    path = ''
    for name in ['results', model_name, dataset_name]:
        path = os.path.join(path, name)
        if not os.path.isdir(path):
            os.mkdir(path)
    return path
        
