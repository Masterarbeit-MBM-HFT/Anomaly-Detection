# 1. User Variables
_datasets_common_path = "./dummy_datasets/"

# 2. Import Libraries

## 2.1 Common
import pandas as pd
from modules.portfolio_counts import portfolio_std, portfolio_hist

## 2.2 Supervised Learning Queries
from modules.supervised_learning_queries import warnings_vin

# 3. Functions

## 3.1 Prepare Master Metadata

def prepare_master_metadata(query, layer_name="std"):
    df_portfolio = spark.sql(query).toPandas()

    if layer_name == "std":
        _delivering_entities_with_anaylsis_names = [str(str(row.analysis_entity).lower() + str(row.delivering_entity)) for row in df_portfolio.itertuples(index=False)]
    elif layer_name == "hist":
        _delivering_entities_with_anaylsis_names = [str(row.delivering_entity) for row in df_portfolio.itertuples(index=False)]

    master_metadata = {
        f"_{layer_name}_delivering_entity": _delivering_entities_with_anaylsis_names,
        f"_{layer_name}_limit": df_portfolio[["rows_to_select"]].apply(lambda x: (int(x[0])), axis=1).tolist()
    }

    return master_metadata

## 3.2 Create separate dummy datasets based on portfolio counts

def create_supervised_dummy_dataset(_master_metadata, _datasets_common_path, anomaly_query, anomaly_type="vin", chunk_size=50000):

    _std_delivering_entity = _master_metadata["_std_delivering_entity"]
    _hist_delivering_entity = _master_metadata["_hist_delivering_entity"]
    _std_limit = _master_metadata["_std_limit"]
    _hist_limit = _master_metadata["_hist_limit"]

    n = len(_std_delivering_entity)

    df_final = pd.DataFrame()

    for i in range(n):
        query = anomaly_query(_std_delivering_entity[i], _hist_delivering_entity[i], _std_limit[i], _hist_limit[i])
        df = spark.sql(query).toPandas()
        
        # Mask the entity with a standardized common name z.B., XY50
        df["entity_name"] = "XY50"

        # Save individual entity
        df.reset_index(drop=True).to_csv(_datasets_common_path + f"supervised_learning/warnings_{anomaly_type}/entities/{_std_delivering_entity[i]}.csv",sep=";", index=False)
        print(f"{anomaly_type} | Individual entity {_std_delivering_entity[i]} saved.")

        # Add to the final dataframe
        df_final = pd.concat([df, df_final])

    # Merged entities
    df_final.reset_index(drop=True).to_csv(_datasets_common_path + f"supervised_learning/warnings_{anomaly_type}/{anomaly_type}_all.csv", sep=";", index=False)
    print(f"{anomaly_type} | Merged entities saved.")

    for i in range(0,len(df_final),chunk_size):
        chunk = df_final.iloc[i:i+chunk_size]
        file_index = i // chunk_size  # gives 0,1,2,...

        # Save individual chunks
        chunk.to_csv(_datasets_common_path + f"supervised_learning/warnings_{anomaly_type}/chunks/part_{file_index}.csv",sep=";", index=False)
        print(f"{anomaly_type} | Chunk_{file_index} saved.")

    return 1

# 4. Data Preparation

## 4.1 Master Metadata

_master_metadata = {}

_master_metadata_std = prepare_master_metadata(portfolio_std.query_std, layer_name="std")
_master_metadata_hist = prepare_master_metadata(portfolio_hist.query_hist, layer_name="hist")
_master_metadata = {**_master_metadata_std, **_master_metadata_hist}

print(_master_metadata)

## 4.2 Supervised Learning Dummy Extracts

### 4.2.1 VIN

_vin_result = create_supervised_dummy_dataset(_master_metadata, _datasets_common_path, anomaly_query = warnings_vin.return_base_query, anomaly_type = "vin", chunk_size = 50000)

# 5. Final
print("Success!")