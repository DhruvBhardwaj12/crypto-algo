# Decision Log

## Data

### 2023-03-24 13:00 UTC — Missing candle in BTCUSDT and ETHUSDT 1h

**Finding:** Both BTCUSDT and ETHUSDT 1h datasets are missing the candle
at `2023-03-24 13:00:00 UTC`. This is confirmed to be a synchronous gap
across symbols, which strongly suggests an exchange-side event, not a
fetcher bug.

**Action taken:** None. The gap is recorded here but not filled,
interpolated, or removed. Filling introduces fake information that can
be silently exploited by a strategy.

**Consequence for downstream code:**
- The feature engine must handle gaps without assuming a rigid hourly grid.
- Strategies must be robust to isolated missing candles.
- Backtests covering this period will see one fewer hour of data. This is
  realistic — a live trader would have faced the same hole.

**Status:** Accepted as-is.
