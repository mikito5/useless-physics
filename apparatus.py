from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))
today = datetime.now(JST).strftime("%Y-%m-%d")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(f"""# Daily Useless Physics

✅ The apparatus ran successfully.

- date: {today}

Next: generate a plot.
""")
