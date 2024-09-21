import os
from flask import jsonify
import pandas as pd
from datetime import datetime

class CSVHelper:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def initialize_csv(self):
        """Initialize the CSV file with headers if it does not exist."""
        if not os.path.isfile(self.csv_file_path):
            df = pd.DataFrame(columns=['name', 'imageSrc', 'gender','datetime', 'skintone', 'facetone'])
            df.to_csv(self.csv_file_path, index=False)

    def append_row(self, name, image_src,gender):
        """Append a new row of data to the CSV file."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_record = {'name': name, 'imageSrc': image_src, 'datetime': now, 'skintone': "", 'facetone': "","gender":gender}

        df = pd.read_csv(self.csv_file_path)
        df = df._append(new_record, ignore_index=True)
        df.to_csv(self.csv_file_path, index=False)


    def read_row(self,name):
        df = pd.read_csv(self.csv_file_path)
        filtered_df = df[df['name'].str.lower() == name.lower()]

        if filtered_df.empty:
            return jsonify({"message": "No matching records found"}), 404
        
        filtered_df.fillna('', inplace=True)

    # Convert the filtered DataFrame to a dictionary
        result = filtered_df.to_dict(orient='records')

        return jsonify(result), 200
    

    def edit_row(self,name,face_tone):
        name_to_filter = name.lower()
        df = pd.read_csv(self.csv_file_path)
        df.loc[df['name'].str.lower() == name_to_filter, 'facetone'] = face_tone
        df.fillna('', inplace=True)

        df.to_csv('data.csv', index=False)
        

    def image_url(self,name):
        name_to_filter = name.lower()
        df = pd.read_csv(self.csv_file_path)
        image_column = "imageSrc"
        filtered_df = df[df['name'].str.lower() == name.lower()]
        if filtered_df.empty:
            return jsonify({"message": "No matching records found"}), 404
        
        filtered_df.fillna('', inplace=True)

        url = filtered_df[image_column].to_list()

        return url







