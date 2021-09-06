datasets_dict = {
    'reflex': {
        'reflex': {
            'name': 'reflex',
            'path': 'datasets/reflex',
            'multilabel': True,
            'train_dir': 'train',
            'val_dir': 'val',
            'num_classes': 7,
            'classes': {
                0: 'background_ring',
                1: 'diffuse_scattering', 
                2: 'ice_ring',
                3: 'loop_scattering',
                4: 'non_uniform_detector',
                5: 'strong_background',
                6: 'artifact'
            }
        }
    },

    'diffranet': {
        'synthetic': {
            'name': 'diffranet_synthetic',
            'path': 'datasets/diffranet/synthetic',
            'multilabel': False,
            'train_dir': 'train',
            'val_dir': 'val',
            'num_classes': 5,
            'classes': {
                0: 'blank',
                1: 'no-crystal',
                2: 'weak',
                3: 'good',
                4: 'strong'
            }
        },
        'real_raw': {
            'name': 'diffranet_real_raw',
            'path': 'datasets/diffranet/real_raw',
            'multilabel': False,
            'train_dir': 'val',
            'val_dir': 'test',
            'num_classes': 2,
            'classes': {
                0: 'no_diffraction',
                1: 'diffraction'
            }
        },
        'real_preprocessed': {
            'name': 'diffranet_real_preprocessed',
            'path': 'datasets/diffranet/real_preprocessed',
            'multilabel': False,
            'train_dir': 'val',
            'val_dir': 'test',
            'num_classes': 2,
            'classes': {
                0: 'no_diffraction',
                1: 'diffraction'
            }
        }
    },
}
