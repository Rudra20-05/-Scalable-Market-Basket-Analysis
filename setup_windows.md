# Windows + VS Code Setup Guide

This guide walks through setting up and running the **Scalable Market
Basket Analysis** project on **Windows 10/11** using **VS Code** and
**PowerShell**. Follow the steps in order.

---

## 1. Install Python

1. Download Python 3.10 or 3.11 from https://www.python.org/downloads/
   (PySpark 3.5.x works most reliably on Python 3.9–3.11; avoid 3.12+ for now).
2. During installation, **check "Add python.exe to PATH"**.
3. Verify in PowerShell:

```powershell
python --version
```

---

## 2. Open the project in VS Code

1. Open VS Code.
2. `File → Open Folder...` → select the `scalable_market_basket_analysis` folder.
3. Open a new terminal: `` Ctrl + ` `` (this opens PowerShell by default on Windows).

---

## 3. Create a virtual environment

In the VS Code PowerShell terminal, from the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If you get an error about execution policies, run this once (as the
> current user, no admin needed) and then re-activate:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

You should see `(venv)` at the start of your terminal prompt once activated.
In VS Code, also select this interpreter: `Ctrl+Shift+P` → "Python: Select
Interpreter" → choose the one inside `.\venv\`.

---

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs PySpark, pandas, numpy, streamlit, plotly, matplotlib, and networkx.

---

## 5. Install Java (required by Spark)

PySpark runs on the JVM, so you need Java installed (Java 8, 11, or 17
all work with PySpark 3.5.x — Java 17 is recommended).

1. Download the **Eclipse Temurin JDK 17 (LTS)** from:
   https://adoptium.net/temurin/releases/
   (choose Windows, x64, .msi installer)
2. Run the installer. During installation, check the option
   **"Set JAVA_HOME variable"** if offered.
3. Verify in a **new** PowerShell window:

```powershell
java -version
```

You should see a version string like `openjdk version "17...`".

---

## 6. Configure JAVA_HOME (if not set automatically)

If `java -version` fails or PySpark complains it cannot find Java:

1. Find your JDK install path, typically something like:
   `C:\Program Files\Eclipse Adoptium\jdk-17.0.x.x-hotspot`
2. Open **"Edit the system environment variables"** (search in Start Menu).
3. Click **Environment Variables** → under **User variables**, click **New**:
   - Variable name: `JAVA_HOME`
   - Variable value: the path from step 1
4. Edit the `Path` variable and add: `%JAVA_HOME%\bin`
5. Close and reopen PowerShell / VS Code, then re-verify with `java -version`.

You can also set it just for the current terminal session (temporary):

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.x.x-hotspot"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

---

## 7. Running PySpark (sanity check)

From the project root, with the virtual environment activated:

```powershell
python -c "from pyspark.sql import SparkSession; s = SparkSession.builder.master('local[*]').getOrCreate(); print(s.range(5).count()); s.stop()"
```

If this prints `5` without errors, Spark is working correctly.

> **Common Windows-specific note:** PySpark sometimes prints a warning
> about `HADOOP_HOME`/`winutils.exe` not being found. For this project
> (which only reads/writes local CSV files through pandas, and uses
> Spark purely for in-memory DataFrame processing and FPGrowth), this
> warning can be safely ignored — it does not affect FP-Growth results.

---

## 8. Running the dataset generator

```powershell
python src\generate_dataset.py --num_transactions 10000
```

This creates/overwrites `data\retail_transactions.csv`. You can generate
larger datasets for scalability testing:

```powershell
python src\generate_dataset.py --num_transactions 50000 --output data\retail_transactions_50k.csv
python src\generate_dataset.py --num_transactions 100000 --output data\retail_transactions_100k.csv
python src\generate_dataset.py --num_transactions 500000 --output data\retail_transactions_500k.csv
```

---

## 9. Running the full analysis pipeline

```powershell
python src\run_pipeline.py
```

This will (re)generate the dataset if missing, preprocess it with Spark,
run FP-Growth, generate association rules and business insights, and
write everything to the `results\` folder.

Optional parameters:

```powershell
python src\run_pipeline.py --num_transactions 50000 --min_support 0.02 --min_confidence 0.3
```

---

## 10. Running the dashboard

```powershell
streamlit run dashboard\app.py
```

This opens a browser tab (usually at `http://localhost:8501`) with the
interactive dashboard. Make sure you've run `run_pipeline.py` at least
once before starting the dashboard.

---

## 11. Running performance/scalability testing

```powershell
python src\performance_test.py
```

By default this tests dataset sizes `[10000, 50000, 100000, 500000]`
(edit `PERFORMANCE_TEST_SIZES` in `config\config.py` to change this).
This can take several minutes, especially for the 500,000-transaction
run. To test a smaller/custom set of sizes:

```powershell
python src\performance_test.py --sizes 10000 50000
```

Results are saved to `results\performance_results.csv` and
`results\performance_analysis.png`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and check "Add to PATH", or use `py` instead of `python`. |
| `JAVA_HOME is not set` | Follow step 6 above. |
| PySpark hangs / very slow to start | First Spark session start is always slower (JVM warm-up); subsequent runs are faster. |
| `Address already in use` for Streamlit | Run `streamlit run dashboard\app.py --server.port 8502`. |
| Dashboard shows "Results not found" | Run `python src\run_pipeline.py` first. |
| `ModuleNotFoundError` | Make sure the venv is activated (`(venv)` visible in prompt) and `pip install -r requirements.txt` completed successfully. |
