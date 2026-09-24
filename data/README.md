# Dataset Folder

`retail_transactions.csv` is a synthetically generated retail transaction
dataset created by `src/generate_dataset.py`. It is provided here at a
default size of **10,000 transactions** so the project runs out of the box.

## Columns

| Column          | Description                                    |
|-----------------|-------------------------------------------------|
| transaction_id  | Unique ID per shopping basket (e.g. T00000001)  |
| customer_id     | Unique ID per customer (e.g. C000042)           |
| timestamp       | Purchase date/time                              |
| product_id      | Unique ID per product (e.g. P001)               |
| product_name    | Product name (e.g. "Bread")                     |
| category        | Product category (e.g. "Grocery")               |
| quantity        | Units purchased in this line item               |
| price           | Unit price (with small random jitter)           |

Each row is one **product line** within a transaction — a single
transaction_id will appear on multiple rows, one per product purchased
in that basket.

## Regenerating / resizing the dataset

```bash
python src/generate_dataset.py --num_transactions 10000
python src/generate_dataset.py --num_transactions 50000 --output data/retail_transactions_50k.csv
python src/generate_dataset.py --num_transactions 100000 --output data/retail_transactions_100k.csv
python src/generate_dataset.py --num_transactions 500000 --output data/retail_transactions_500k.csv
```

The generator intentionally biases certain products to co-occur (e.g.
Bread → Butter, Pasta → Pasta Sauce, Coffee → Sugar) so that FP-Growth
has genuine patterns to discover — but this bias is only used during
*generation*; the mining algorithm rediscovers these relationships from
scratch. A small amount (~0.5%) of realistic data-quality noise
(duplicates, missing product names, invalid quantities/prices) is also
injected, which `src/preprocess.py` cleans up.
