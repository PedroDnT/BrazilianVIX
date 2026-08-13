# Brazilian VIX

A VIX-style forward-looking volatility index for the Ibovespa, computed from B3 option
chains. B3 publishes no VIX equivalent, so the index is built from scratch and validated
against realised volatility.

## Constraint

The CBOE VIX methodology assumes inputs that B3's listed option market does not supply
cleanly. Adapting it meant choosing, at each step, between theoretical fidelity and
producing a signal at all from the strikes actually quoted. The code favours the latter
and states where that trade-off was made.

## Data

Every run fetches over HTTP and holds results in memory. **There is no database.**

| Source | Used for | Code |
|---|---|---|
| OpLab REST API (`api.oplab.com.br/v3`) | Option chains, interest rates | `oplab_api.py` — `fetch_options_data()`, `fetch_interest()`, `get_historical_options()` |
| BRAPI (`brapi.dev/api`) | Prime rate | `BrAPIWrapper.py` |
| yfinance (`^BVSP`) | Ibovespa closes, for realised volatility | `VolCalculation.ipynb` |

Requires `OPLAB_API_KEY` in the environment.

## Method

1. Pull the IBOV option chain and the prevailing interest rate.
2. Select strikes and expiries adapted to what is actually quoted.
3. Compute a daily forward-looking volatility level.
4. Compute realised volatility from Ibovespa closes and compare.

## Result

Over the sample in `VolCalculation.ipynb`, the index and realised volatility have a
Pearson correlation of **0.7531** — the index tracks the direction of realised
volatility.

**This is a co-movement result, not a calibration result.** The two series are not on a
comparable scale: the index ranges roughly 1194–3189 while realised volatility ranges
5.22–15.05, and the comparison cell applies a hand-tuned `× 50` rescale purely to overlay
them on one chart. The correlation is invariant to that rescale, so the 0.7531 stands on
its own — but the index is not yet expressed in units that can be read as a volatility
percentage, and no RMSE, R² or regression fit is computed. Levels should not be compared
across the two series until the scaling is derived rather than fitted by eye.

## Known issues

- **Scaling is unresolved.** The index is not in interpretable volatility units (see above).
  The `× 50` factor sits under a comment that says 100 — neither figure is derived.
- **`oplab_api.py:675` is broken.** It does `from Brapi import BrAPIWrapper`; the module is
  `BrAPIWrapper.py`, so `calculate_vix_df` raises `ModuleNotFoundError` on that path.
- The notebook was executed out of order across sessions, so stored execution counts are
  not sequential.

## Running

```bash
pip install -r requirements.txt
export OPLAB_API_KEY=...
jupyter notebook VolCalculation.ipynb
```

## License

MIT. See [LICENSE](LICENSE).

---

Pedro Todescan
