"""
This file contains all information necessary for processing datasets.
datasets_dict keys:
:main key: dataset group name, it collects all subdatasets for this dataset
:subkeys: subdatasets names
subsubkeys:
:name (str): subdataset name,
:path (str): relative path to the dataset,
:multilabel (bool): if True an image can belong to more than one class,
:train_dir: (str): a relative path to train dataset,
:val_dir: (str): a relative path to val dataset,
:test_dir: (str): a relative path to test dataset,
:num_classes: (int): total no. of classes,
:'classes': (dict): keys - class names, values - class indices
"""


datasets_dict = {
    'ln84': {
        'ln84_raw': {
            'name': 'ln84_raw',
            'path': 'datasets/ln84_raw',
            'multilabel': False,
            'train_dir': 'train',
            'val_dir': 'val',
            'test_dir': 'val',
            'num_classes': 2,
            'classes': {
                'miss': 0,
                'hit': 1
            }
        },
        'ln84_preprocessed': {
            'name': 'ln84_preprocessed',
            'path': 'datasets/ln84_preprocessed',
            'multilabel': False,
            'train_dir': 'train',
            'val_dir': 'val',
            'test_dir': 'val',
            'num_classes': 2,
            'classes': {
                'miss': 0,
                'hit': 1
            }
        }
    },

    'reflex': {
        'reflex': {
            'name': 'reflex',
            'path': 'datasets/reflex',
            'multilabel': True,
            'train_dir': 'train',
            'val_dir': 'val',
            'test_dir': 'test',
            'num_classes': 7,
            'classes': {
                'background_ring': 0,
                'diffuse_scattering': 1, 
                'ice_ring': 2,
                'loop_scattering': 3,
                'non_uniform_detector': 4,
                'strong_background': 5,
                'artifact': 6
            }
        }
    },

    'diffranet': {
        'synthetic': {
            'name': 'diffranet_synthetic',
            'path': 'datasets/diffranet_synthetic',
            'multilabel': False,
            'train_dir': 'train',
            'val_dir': 'val',
            'test_dir': 'test',
            'num_classes': 5,
            'classes': {
                'blank': 0,
                'no-crystal': 1,
                'weak': 2,
                'good': 3,
                'strong': 4
            }
        },
        'real_raw': {
            'name': 'diffranet_real_raw',
            'path': 'datasets/diffranet_real_raw',
            'multilabel': False,
            'train_dir': 'val',
            'val_dir': 'test',
            'test_dir': 'test',
            'num_classes': 2,
            'classes': {
                'no_diffraction': 0,
                'diffraction': 1
            }
        },
        'real_preprocessed': {
            'name': 'diffranet_real_preprocessed',
            'path': 'datasets/diffranet_real_preprocessed',
            'multilabel': False,
            'train_dir': 'val',
            'val_dir': 'test',
            'test_dir': 'test',
            'num_classes': 2,
            'classes': {
                'no_diffraction': 0,
                'diffraction': 1
            }
        }
    },
}
