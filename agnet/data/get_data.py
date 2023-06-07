import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from data import AGDataset

def get_train_test_dl(df, train_idx, test_idx, image_base_path= "./", target_fields = ['age', 'gender'], batch_size=32, workers=1, *args, **kwargs):
    # split dataframe in train and eval set
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    train_dl = DataLoader(
        AGDataset(train_df, base_path=image_base_path, target_field=target_fields),
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=True,
        num_workers=workers
    )

    test_dl = DataLoader(
        AGDataset(test_df, base_path=image_base_path, target_field=target_fields),
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=True,
        num_workers=workers
    )
    return train_dl, test_dl

def prep_dataloader(file_path, config):
    df = pd.read_csv(file_path)
    spliter = StratifiedShuffleSplit(n_splits=config['n_splits'], train_size=config['train_size'], random_state=config['random_state'])
    for train_idx, test_idx in spliter.split(df, df[config['target_fields']]):
        return get_train_test_dl(
            df, train_idx, test_idx, **config
        )
