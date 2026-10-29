# Everesteer Hedge Fund Hackathon

My entry for the Everesteer hackathon at LSE (29 October 2026): a live prediction tournament where you predict forward returns on futures baskets, then stake money on your own models over four 20-minute rounds. The final stake balance decides the winner.

**Result:** $20.00 became $18.25. The reason it went wrong turned out to be the most useful part, so the full post-mortem is in [REPORT.md](REPORT.md).

## The short version

- I built recent-window LightGBM models with auxiliary-target blends, and picked the training window and the blend on the practice board. The practice board covers a single period, and choosing among about 25 models on one period overfits: practice scores of +0.20 became -0.07 in round 1.
- A forward test across four separate time blocks showed the 1/10 window was too short. Every variant that used more data or more regularisation beat it, and the last quarter of history won all four blocks.
- The live rounds agreed. Over all four rounds the best model trained on full history with recent periods weighted up; over the last two rounds the best was the last-quarter model, our only positive model in round 3.
- My biggest loss came from abandoning that evidence: in round 3 I concentrated the whole stake on round 2's two most volatile models to take more risk, and lost $2.60.

## How the scoring works

Each round is scored as `S = FIT + 3*UNQ + INOV - (benchmark FIT + INOV)`, squashed with an arctan. FIT is the correlation with the target, UNQ is the part not explained by the event's benchmark model (it counts three times), and INOV is measured against a hidden core feature set. Payout is stake times score. A model that copies the benchmark scores zero, and one that looks like a weaker copy of it scores negative, which is what happened to mine in the losing rounds.

## Running it

The training data is not in this repo. Clone Everesteer's starter repo, [everestquant/hackathon-example-scripts](https://github.com/everestquant/hackathon-example-scripts), into a `hackathon-example-scripts/` folder here: the scripts read `futures_train.parquet` and the benchmark file from there. The event API key goes in the `EIQ_API_KEY` environment variable (or a profile file outside the repo, see `eiq.py`); keys are issued per event and stop working once it ends.

```
pip install -r requirements.txt
python explore.py aux4,aux4_w4,aux4_w6     # 4-block forward test against aux4
python candidates.py final C_aux_w4        # fit a candidate on the true last window
python submit_round.py --only bl-aux-w4    # predict the open round and submit
python stake_clusters.py round_1,round_2   # show a stake plan (add --apply to stake)
```

## Files

| File | What it does |
|---|---|
| `explore.py` | The forward test: fits each idea on a window that ends 20 periods before each of 4 test blocks at the end of train, then scores every block |
| `hosted_folds.py` | The same test for models trained on Everesteer's hosted compute (launch, collect, evaluate) |
| `candidates.py`, `train_*.py` | Final fits on the most recent windows, saved into one model file |
| `submit_round.py` | Recipes (model name to blend), the round lineup, prediction and upload |
| `round_watch.py` | Waits for a round to open, then submits the lineup once |
| `stake_clusters.py` | Stakes equally per cluster of correlated models, with a small tilt from live scores |
| `research.py` | The first holdout study (benchmark-residualised models, which turned out to be a trap) |
| `gauge.py`, `period_analysis.py` | A pre-stake risk gauge based on how dispersed a round's features are |
| `board.py`, `stake.py` | Leaderboard and staking position |

Built with Claude Code. I set the direction and made the calls on testing and staking; the reasoning behind each one, including the ones that cost money, is in the report.
