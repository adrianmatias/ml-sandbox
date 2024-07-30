# ragblog

This project perfomrs Retrieval Augmented Generation on from a crawled blog.

## Init

### Install poetry environment
```
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"
poetry install
```
### Lint
```
sh lint.sh
```
### Test
```
pytest -vv
```
### run
```
cd ragblog
python run.py
```



