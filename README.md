# convmf
code for "Convolution Matrix Factorization for Document Context-Aware Reccomendation"


# Setup
* install python module.
```bash
pip install -r requirements.txt
```

* download rating data
```bash
./download_data.sh
```

* download movie description data
  * visit https://www.kaggle.com/tmdb/tmdb-movie-metadata
  * download data and expand zip into ./data/
  * data directory must be as follows:
  ```bash
    - data/
      - ml-10M100K/
      - tmdb-5000-movie-dataset/
  ```
  
# Run
```bash
python preprocessing.py
```

```bash
python train.py
```