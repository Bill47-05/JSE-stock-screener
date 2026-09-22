# JSE Stock Screener

Clean rebuild of your multi-indicator JSE screening app.

## Features
- Liquidity filter (≥ 10 000 avg volume)
- Price filter (R10 – R120)
- Ichimoku (price above cloud + Tenkan > Kijun)
- Alligator (Lips > Teeth > Jaw and rising)
- MACD > Signal
- RSI(14) 50–75
- CCI(20) > 0

- Recommendations table
- Interactive review charts
- Portfolio tracker (buy / sell / average price / P&L)
- iPad optimised

## Deploy (Streamlit Community Cloud)
1. Upload this folder to a public GitHub repo
2. Go to https://share.streamlit.io
3. Create app → select repo → main file `app.py` → Deploy
4. On iPad: open the URL in Safari → Share → Add to Home Screen

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```
