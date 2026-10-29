# Everesteer Hedge Fund Hackathon

My entry for the Everesteer hackathon at LSE (29 October 2026): a live prediction tournament where you predict forward returns on futures baskets, then stake money on your own models over four 20-minute rounds, each scored against a sealed answer key you never see.

## Highlights

- **2nd of 521 in live round 2.** My bagged LightGBM model (`bl-c-bag`) scored +0.135 on the round's sealed answer key, and six of my models finished in that round's top 14 (2nd, 5th, 9th, 10th, 11th and 14th).
- **2nd of 174 on the practice board** before the live rounds, with a recent-window LightGBM, at the time I submitted it.
- **Every model positive in a tough round.** In round 4 all 16 of my models scored positive while the median entry was negative; the best placed 20th of 598.
- **A forward-testing harness built mid-event** that tests every idea on four separate time blocks, and a hosted-compute pipeline that ran 84 training jobs on Everesteer's servers and scored them with the same test.

Round-by-round results, the testing and the staking decisions are in [REPORT.md](REPORT.md).

## What I built

- **Models:** LightGBM on recent windows of history, blended with models trained on auxiliary targets (related return series the data provides alongside the graded one), plus CatBoost, XGBoost and ridge variants for diversity.
- **Forward test (`explore.py`):** each idea is fitted on a window that ends 20 periods before a test block and scored on that block, for four blocks at the end of the training data. An idea has to beat the baseline on at least three of the four to go live.
- **Hosted training (`hosted_folds.py`):** launches the same test on Everesteer's compute, throttled to the credit hold, then pulls each fitted model back for local scoring.
- **Live operations:** a watcher that submits the whole lineup the moment a round opens, and a staking script that spreads money equally across clusters of correlated models.

## What I learned

- **Test on many periods, not one.** The practice board is a single period, and the round-to-round ordering of my models flipped more than once. The four-block forward test was the evidence that held up: it found that training on the last quarter of history beat the shorter window I had started with, and the live rounds agreed.
- **Similar models are one bet.** My models rose and fell together against the field, so splitting the stake between them diversified less than it looked. The next step is a second signal built a different way.

## How the scoring works

Each round is scored as `S = FIT + 3*UNQ + INOV - (benchmark FIT + INOV)`, squashed with an arctan. FIT is the correlation with the target, UNQ is the part not explained by the event's benchmark model (it counts three times), and INOV is measured against a hidden core feature set. Payout is stake times score, so a model has to beat the benchmark, not just predict well.

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

Built with Claude Code. I set the direction and made the calls on testing and staking.
