import os
from sklearn.model_selection import train_test_split

labels_txt_path = 'datasets/ln84_labels.txt'
root_source_path = 'datasets/images2/'
root_target_path = 'datasets/'

root_source_path_raw = os.path.join(root_source_path, 'images')
root_source_path_preprocessed = os.path.join(root_source_path, 'images_p')

root_target_path_raw = os.path.join(root_target_path, 'ln84_raw')
root_target_path_preprocessed = os.path.join(root_target_path, 'ln84_preprocessed')

def split_ln84():
    """Split ln84 into train and val datasets.
    Train and val consist of images from "MISS" and "HIT" classes.
    Images from class "MAYBE" are stored in a separate folder - "2"
    for raw and preprocessed respectively.
    ----------------------------------------------------------------
    Target directory structure:
    {ln84_raw, ln_84_preprocessed}
    |_{train, val}
      |_0
      |_1
    |_2
      |_2
    """
    img_names, labels = [], [] # MISS or HIT
    img_names_2, labels_2 = [], 2 # MAYBE

    with open(labels_txt_path, 'r') as f:
        # Get label and filename from each line of a txt file with annotations
        # Example of a line: 'r0095_2000.h5 [1] MISS'
        # Number in square brackets indicates the filename
        # Example of a filename: '00619.png'
        for line in f:
            img_name = line[line.find("[")+1:line.find("]")] # Get label from inside of the brackets
            img_name = img_name.zfill(5) # Fill string with zeros to match img naming convention
            img_name = ''.join([img_name, '.png']) # Add img extension to img name
            if line.find('MISS')!=-1: # If MISS in line, img belongs to class 0
                label = '0'
            elif line.find('HIT')!=-1: # If HIT in line, img belongs to class 1
                label = '1'
            else:
                img_names_2.append(img_name) # Else (if MAYBE) in line, img belongs to class 2
                continue
            img_names.append(img_name)
            labels.append(label)

    # Train-val split
    # Only imgs from class zero and 1 are included
    imgs_train, imgs_val, labels_train, labels_val = train_test_split(img_names, labels, test_size=0.8, shuffle=True, stratify=labels)
    datasets = {}
    datasets['train'] = zip(imgs_train, labels_train)
    datasets['val'] = zip(imgs_val, labels_val)

    # Imgs from class 2 are stored separately
    labels_2 = ['2']*len(img_names_2)
    datasets['2'] = zip(img_names_2, labels_2)

    # Move imgs from source path to target path
    for dataset_name, dataset in datasets.items():
        for img_name, label in dataset:
            in_path_raw = os.path.join(root_source_path_raw, img_name)
            out_path_raw = os.path.join(root_target_path_raw, dataset_name, label, img_name)
            in_path_preprocessed = os.path.join(root_source_path_preprocessed, img_name)
            out_path_preprocessed = os.path.join(root_target_path_preprocessed, dataset_name, label, img_name)

            # os.replace(in_path_raw, out_path_raw)
            # os.replace(in_path_preprocessed, out_path_preprocessed)

if __name__ == '__main__':
    split_ln84()