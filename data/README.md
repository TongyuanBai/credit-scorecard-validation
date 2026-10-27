# Data

The data are not stored in this repository. Run, from the repository root,

```bash
python scripts/download_data.py
```

which writes `data/lending_club.csv` and verifies its checksum. Another folder can be used by setting the environment variable `SCORECARD_DATA_DIR`.

| Item | Value |
|---|---|
| Data set | `lending_club` from the R package [`modeldata`](https://modeldata.tidymodels.org/reference/lending_club.html) |
| Download | [Rdatasets](https://github.com/vincentarelbundock/Rdatasets) copy: `csv/modeldata/lending_club.csv` |
| Content | 9,857 Lending Club loans issued in the first quarter of 2016, 22 predictors and the outcome `Class` |
| Outcome | `bad`: charged off, in default or 21-120 days late when the data were extracted; `good` otherwise (5.25% bad) |
| SHA-256 | `0e27aa7be8f496fde17083a7ca28487de84ee0e66da6fe09c0d69f19d5adcb7f` |

The file has no application or extraction dates. A data dictionary for the variables is available from Lending Club's published loan statistics; the descriptions used in the report are in `scorecard/data.py`.
