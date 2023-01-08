# arxiv-digest-explorer

CLI tool for exploring arXiv digest that collect data for future automation.

## Usage

Copy the arXiv email digest into a file and run `explore.py` on it. The script
will guide you through the abstract and open the you are going to read in a
browser.

The activity is logged into file `positive.jsonl` and `negative.jsonl`. The
logs will be later used to train an automatic scorer for selecting papers I am
more likely to read.
