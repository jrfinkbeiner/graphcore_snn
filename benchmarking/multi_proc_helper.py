dataset = None

def set_global_dataset(init_dataset):
    global dataset
    dataset = init_dataset

def get_dataset_item(idx):
    return dataset[idx]