#!/usr/bin/env bash
# Fetch the StockNet dataset (~580 MB). Not committed to this repo.
#
#   Source   : https://github.com/yumoxu/stocknet-dataset
#   License  : MIT (see LICENSE inside the cloned repo)
#   Citation : Xu & Cohen, "Stock Movement Prediction from Tweets and
#              Historical Prices", ACL 2018
#
# Expected layout after running:
#   data/external/stocknet-dataset/price/raw/{TICKER}.csv
#   data/external/stocknet-dataset/tweet/raw/{TICKER}/{YYYY-MM-DD}
#   data/external/stocknet-dataset/StockTable
set -euo pipefail

DEST="data/external/stocknet-dataset"
mkdir -p data/external

if [ -d "$DEST" ]; then
  echo "already present at $DEST -- skipping"
  exit 0
fi

echo "cloning StockNet (~580 MB, this takes a few minutes)..."
git clone --depth 1 https://github.com/yumoxu/stocknet-dataset.git "$DEST"

echo
echo "price files : $(ls "$DEST/price/raw" | wc -l)"
echo "tweet dirs  : $(ls "$DEST/tweet/raw" | wc -l)"
echo "done -> $DEST"
