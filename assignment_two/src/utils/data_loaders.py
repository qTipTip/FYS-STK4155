from pathlib import Path
from typing import Tuple

import pandas
import torch
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader


class CreditCardData(Dataset):

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.num_items

    def __init__(self,
                 file_path: str = '/home/ivar/Downloads/FYS-STK4155/assignment_two/data/credit_card_data.xls') -> None:
        super().__init__()

        self.file = Path(file_path)
        self.fetch_data()

    def fetch_data(self) -> None:
        """
        Responsible for reading the credit card data-file.
        """

        if not self.file.exists():
            raise FileNotFoundError()

        data = pandas.read_excel(self.file.absolute(), header=1, skiprows=0, index_col=0)
        data.rename(index=str, columns={"default payment next month": "default_payment_next_month"}, inplace=True)

        data = self.remove_zero_rows(data)
        X, y = self.extract_and_transform_data(data)

        self.X = X
        self.y = y

        self.num_items = self.X.shape[0]
        self.num_features = self.X.shape[1]

    def extract_and_transform_data(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the data and performs normalization.
        :param data: pandas dataframe containing the credit card data.
        :return: input variables X and response variables y.
        """

        X = data.loc[:, data.columns != 'default_payment_next_month'].values
        y = data.loc[:, data.columns == 'default_payment_next_month'].values

        # Onehot-encoding.
        # TODO: Make this work with sparse pytorch tensors.
        one_hot_enc = OneHotEncoder(categories='auto', sparse=False)

        # We set remainder to pass-through to avoid dropping untransformed columns
        # One-hot-encodes the categorical columns, corresponding to GENDER, EDUCATION, MARITAL_STATUS
        X = ColumnTransformer([("", one_hot_enc, [1, 2, 3]), ], remainder='passthrough').fit_transform(X)
        # y = one_hot_enc.fit_transform(y)
        # The CrossEntropyLoss expects class-labels and not one-hot encoded targets, hence we do not one-hot-encode,
        # and rather squeeze the array to get conforming shapes.
        y = y.squeeze()

        # Input scaling
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        x_mean = X.mean(0, keepdim=True)
        x_stdev = X.std(0, unbiased=False, keepdim=True)
        X -= x_mean
        X /= x_stdev

        return X, y

    @staticmethod
    def remove_zero_rows(data):
        """
        Removes any rows where the data is identically zero across all payments.
        :param data:
        :return: data with dropped rows
        """

        data = data.drop(data[(data.BILL_AMT1 == 0) &
                              (data.BILL_AMT2 == 0) &
                              (data.BILL_AMT3 == 0) &
                              (data.BILL_AMT4 == 0) &
                              (data.BILL_AMT5 == 0) &
                              (data.BILL_AMT6 == 0)].index)

        data = data.drop(data[(data.PAY_AMT1 == 0) &
                              (data.PAY_AMT2 == 0) &
                              (data.PAY_AMT3 == 0) &
                              (data.PAY_AMT4 == 0) &
                              (data.PAY_AMT5 == 0) &
                              (data.PAY_AMT6 == 0)].index)

        return data


if __name__ == '__main__':
    D = CreditCardData()
