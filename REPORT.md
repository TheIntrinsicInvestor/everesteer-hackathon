# Post-mortem: what went wrong, and what the evidence said

## The event

Four 20-minute rounds on a shared clock. Each round serves a fresh "live" set of about 19,500 rows over 217 periods with the targets blanked; training data stays the same all event. You upload predictions for as many models as you like, stake on up to 20 of them (minimum $1 each), and each staked model pays stake times its round score. Scores are relative to Everesteer's benchmark model, whose own FIT and INOV are subtracted, and UNQ (what your model gets right that the benchmark does not) counts three times.

## Result

| Round | Staked | P&L | Field median score | What happened |
|---|---|---|---|---|
| 1 | $17.00 | -$1.20 | +0.030 | Every model about -0.07: positive FIT, negative UNQ |
| 2 | $15.00 | +$0.95 | +0.055 | Every model positive; an unstaked one ranked 2nd of 521 |
| 3 | $19.68 | -$2.60 | +0.077 | Concentrated stake lost; only one model positive |
| 4 | $17.14 | +$1.10 | -0.038 | Every model positive while the field went negative |

Final balance $18.25 from $20.00.

## What went wrong in round 1

All eleven models had positive FIT (0.009 to 0.014) and positive INOV, so they were predicting the right direction. They lost because the benchmark had a strong round: working back from the score formula, its FIT plus INOV was about +0.041 against our +0.018, and our UNQ was slightly negative. A model that resembles the benchmark but is weaker loses on both counts.

The deeper cause was overfitting, in two forms.

1. **Model selection.** Before round 1 I compared about 25 models on the practice board and picked the best: a LightGBM trained on the most recent 1/10 of history, blended with models trained on auxiliary targets. The practice board is a single period. Picking the winner of 25 on one period selects for luck, and the scores fell from +0.20 on practice to -0.07 live. Round 2 then reversed round 1's window ordering, which is what noise looks like.
2. **The models themselves.** The same LightGBM fitted on the 1/10 window scored 0.160 FIT on its own training rows against -0.006 in cross-validation. It was memorising.

## The forward test

To stop choosing on one period, every idea went through the same test (`explore.py`): fit on a training window that ends 20 periods before a test block, score that block, and repeat for four separate blocks at the end of the training data. An idea had to beat the baseline (aux4, the 1/10-window aux blend) in at least three of the four blocks.

| Idea | Blocks won vs aux4 | Mean FIT |
|---|---|---|
| aux4 on the last 1/4 of history | 4 of 4 | +0.0056 |
| aux4 on periods whose features resemble the test block | 3 | +0.0056 |
| aux4 on the last 1/2 | 3 | +0.0037 |
| aux4 on the last 1/6 | 4 | +0.0036 |
| Blend of the 1/10 and 1/4 windows | 3 | +0.0029 |
| Heavier regularisation (7 leaves, 1,000 rows per leaf) | 4 | +0.0023 |
| Weakly related auxiliary targets only | 3 | +0.0014 |
| Hosted XGBoost and CatBoost on those targets | 2 | +0.0011 |
| aux4 plus the weakly related targets | 4 | +0.0005 |
| Three short windows blended (1/12, 1/10, 1/8) | 3 | -0.0001 |
| Dropping the least stable features | 2 | -0.0001 |
| **aux4 (baseline, 1/10 window)** | | **-0.0005** |
| Mild regularisation | 1 | -0.0009 |
| Hosted MLP on the auxiliary targets | 1 | -0.0012 |
| Hosted MLP, five seeds averaged | 1 | -0.0036 |

The pattern is consistent: almost everything that reduced variance, by using more data or regularising harder, beat the 1/10 window. "Recency is the edge" had come from the single-period practice board.

The hosted batch was useful mainly as a screen. Two practical notes: the hosted MLP preset is scikit-learn, so it runs on CPU and a GPU tier changes nothing but the price; and the delivered model files are pickled under Python 3.11, so under 3.12 the wrapper fails and the fitted estimator has to be pulled out of it (`hosted_folds.py` does this).

## What the live rounds said

Every submitted model was scored every round, staked or not, so the live record covers all of them. Two comparisons are fair, because models added later only share some rounds:

| Model | R1 | R2 | R3 | R4 | Mean |
|---|---|---|---|---|---|
| Last 1/4 of history (added round 3) | | | +0.030 | +0.064 | +0.047 |
| Full history, recent periods weighted up | -0.083 | +0.081 | -0.028 | +0.084 | +0.013 |
| Cross-family blend | -0.069 | +0.115 | -0.100 | +0.097 | +0.011 |
| Bagged 1/10 window | -0.101 | +0.135 | -0.137 | +0.091 | -0.003 |
| My main staked blend (round 1) | -0.068 | +0.089 | -0.149 | +0.075 | -0.013 |
| 1/12 window | -0.071 | +0.057 | -0.166 | +0.051 | -0.032 |

Over all four rounds the best model used full history with recent periods weighted up. Over rounds 3 and 4, where all sixteen models ran, the best was the last-quarter model. Both use more history than the 1/10 window, which matches the forward test. The caveat is size: four rounds is very little, and most gaps between models are smaller than the round-to-round swing.

## Staking

- **Round 1:** $17 of $20, mostly on the main blend. Concentrated because the models correlated 0.75 to 0.89, and a bet on practice-board evidence that did not hold.
- **Rounds 2 and 3:** after the round 1 loss I switched to equal stakes per cluster of correlated models, with live scores allowed to tilt a cluster by at most $1. This follows DeMiguel, Garlappi and Uppal's "1/N" result: with little data, estimated optimal weights do not beat equal weights out of sample. Every candidate was still submitted each round, so unstaked models were free live trials.
- **Round 3 change of plan:** only the top 3 are paid, so I put the whole balance on round 2's two most volatile models to take more risk. Both scored about -0.13: -$2.60. That was acting on one round's winners, the same mistake as round 1 in a different form.
- **Round 4:** the full balance went on the last-quarter model as a test of the main finding. It made +$1.10.

## Lessons

1. **Only multi-period tests held up.** The practice board and individual live rounds kept reversing; the four-block forward test pointed the same way throughout, and the live record ended up agreeing with it.
2. **The window was too short.** More history, or recent periods weighted up over full history, beat the most recent 1/10.
3. **My sixteen models were one bet.** Against the field they rose and fell together every round, so splitting the stake between them never diversified much. Real diversification needs a second signal built differently, and the negative UNQ in the losing rounds says the same thing.
