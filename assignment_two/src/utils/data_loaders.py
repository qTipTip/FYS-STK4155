from pathlib import Path
from typing import Tuple

import numpy as np
import pandas
import torch
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, normalize, scale, PolynomialFeatures
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
        X = scale(X)
        X = torch.tensor(X, dtype=torch.double)
        y = torch.tensor(y, dtype=torch.long)

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


class FrankeDataSet(Dataset):

    def __init__(self, num_points, signal_to_noise=0.1, noisy=True, random_seed=None, degree=5):
        self.num_points = num_points
        self.random_seed = random_seed
        self.stn = signal_to_noise
        self.noisy = noisy
        self.degree = degree
        self.polyfit = PolynomialFeatures(degree)
        if random_seed:
            np.random.seed(random_seed)

        self._sample()

    def _sample(self):
        self.xy = np.random.uniform(0, 1, (self.num_points, 2))
        self.X = self.polyfit.fit_transform(self.xy)
        self.z = np.array([self._eval(x, y) for x, y in self.xy]).reshape((self.num_points, 1))

        if self.noisy:
            self.z_noise = self.z + np.random.normal(0, 1, size=self.z.shape) * self.stn

    def _eval(self, x, y):
        a = 3 / 4 * np.exp(-(9 * x - 2) ** 2 / 4 - (9 * y - 2) ** 2 / 4)
        b = 3 / 4 * np.exp(-(9 * x + 1) ** 2 / 49 - (9 * y + 1) / 10)
        c = 1 / 2 * np.exp(-(9 * x - 7) ** 2 / 4 - (9 * y - 3) ** 2 / 4)
        d = 1 / 5 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

        return a + b + c - d

    def __len__(self):
        return self.num_points

    def __getitem__(self, item):
        if self.noisy:
            return self.xy[item], self.z_noise[item]
        else:
            return self.xy[item], self.z[item]


if __name__ == '__main__':
    D = CreditCardData()
    F = FrankeDataSet(num_points=10000)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    axs = Axes3D(fig)

    axs.scatter(F.xy[:, 0], F.xy[:, 1], F.z_noise)
    plt.show()
