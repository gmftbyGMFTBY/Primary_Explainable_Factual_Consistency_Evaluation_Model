# Generate the dataset for training the explainable evaluation Model

1. process the esnli data split

```bash
python process_esnli.py
```

2. process the ficle dataset

```bash
python process_ficle.py
```

3. process the fever dataset

combining the ficle data split
```bash
python process_fever.py
```

4. combine all the resources

```bash
# combine alpaca, esnli, fever data split, generate train.json
python combine.py
```
