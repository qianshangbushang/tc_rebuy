import pandas as pd
from pydantic import BaseModel

class Dataset(BaseModel):
    test_f1: pd.DataFrame
    train_f1: pd.DataFrame
    user_f1: pd.DataFrame
    log_f1: pd.DataFrame
    test_f2: pd.DataFrame
    train_f2: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


def load_data():
    """Load data from a file."""

    df_test_f1 = pd.read_csv("../data/format1/data_format1/test_format1.csv")
    df_train_f1 = pd.read_csv("../data/format1/data_format1/train_format1.csv")
    df_user_f1 = pd.read_csv("../data/format1/data_format1/user_info_format1.csv")
    df_log_f1 = pd.read_csv("../data/format1/data_format1/user_log_format1.csv")
    df_test_f2 = pd.read_csv("../data/format2/data_format2/test_format2.csv")
    df_train_f2 = pd.read_csv("../data/format2/data_format2/train_format2.csv")

    return Dataset(
        test_f1=df_test_f1,
        train_f1=df_train_f1,
        user_f1=df_user_f1,
        log_f1=df_log_f1,
        test_f2=df_test_f2,
        train_f2=df_train_f2,
    )


def load_dataframe():
    df_test_f2 = pd.read_csv("../data/format2/data_format2/test_format2.csv")
    df_train_f2 = pd.read_csv("../data/format2/data_format2/train_format2.csv")
    return pd.concat([df_test_f2, df_train_f2], axis=0)