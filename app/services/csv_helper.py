import os
import pandas as pd
from datetime import datetime

class CSVHelper:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def initialize_csv(self):
        """Initialize the CSV file with headers if it does not exist."""
        if not os.path.isfile(self.csv_file_path):
            df = pd.DataFrame(columns=['name', 'imageSrc', 'datetime', 'skintone', 'facetone'])
            df.to_csv(self.csv_file_path, index=False)

    def append_row(self, name, image_src):
        """Append a new row of data to the CSV file."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_record = {'name': name, 'imageSrc': image_src, 'datetime': now, 'skintone': "", 'facetone': ""}

        df = pd.read_csv(self.csv_file_path)
        df = df._append(new_record, ignore_index=True)
        df.to_csv(self.csv_file_path, index=False)


    def read_row(self):
        df = pd.read_csv(self.csv_file_path)
        data = df[df['name'] == "karthee"]
        return data
