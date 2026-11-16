# Data Installation

## 1. Organize directory structure

We recommend the following project structure:

```
data/
├── processed_wide/
    ├───train.csv
    └── val_test.csv
```

Then,
```
mkdir -p data/processed_wide 
```

## 2. Download datasets

Download the required datasets and ground truth file:

You can download them using `curl -L`:

```bash
cd data/processed_wide/
curl -L https://drive.google.com/file/d/1FPY_KRk7VQ8TcCg4smkrD9ytKgB3QI7X/ -o train.csv
curl -L https://drive.google.com/file/d/1IdR5PqLZz4t4cvmFXm9Z8gRpPDISgGKa/ -o val_test.csv
```

