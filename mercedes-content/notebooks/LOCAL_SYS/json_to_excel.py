# 0. User Variables

input_json_path = "../data/DQ_Configuration.json"
output_excel_path = "../data/DQ_Configuration.xlsx"

#######################################################################################################################

# 1. Import Libraries

import pandas as pd
import json

# 2. Functions

# Recursive row-expanding flattener
def expand_records(record, parent_data=None):
    if parent_data is None:
        parent_data = {}
    
    rows = []
    for key, value in record.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    # Merge current non-list data into each sub-item
                    sub_parent = {**parent_data, **{k: v for k, v in record.items() if not isinstance(v, list)}}
                    rows.extend(expand_records(item, sub_parent))
                else:
                    sub_parent = {**parent_data, **record}
                    rows.append(sub_parent)
        elif isinstance(value, dict):
            rows.extend(expand_records(value, {**parent_data, **{k: v for k, v in record.items() if k != key}}))
    
    if not any(isinstance(v, (list, dict)) for v in record.values()):
        rows.append({**parent_data, **record})
    
    return rows

# 3. File Handling

# Read JSON file
with open(input_json_path, "r") as f:
    data = json.load(f)

# Flatten each top-level entry into rows
all_rows = []
for entry in data:
    all_rows.extend(expand_records(entry))

# Create DataFrame with consistent columns
df = pd.DataFrame(all_rows)

# Create an excel file
df.to_excel(output_excel_path)