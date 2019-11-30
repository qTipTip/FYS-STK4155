from pathlib import Path

import pandas
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


class CreditCardData(Dataset):

    def __getitem__(self, index: int) -> T_co:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.num_items

    def __init__(self, file_path: str = '../../data/credit_Card_data.xls') -> None:
        super().__init__()

        self.file = Path(file_path)
        self.fetch_data()

    def fetch_data(self):
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
