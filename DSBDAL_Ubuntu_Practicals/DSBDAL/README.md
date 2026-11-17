# DSBDAL Practical Repository (Ubuntu)

This repository follows the 12 experiments in the college-provided **List of Experiments DSBDA**.

## 1. Ubuntu setup

Open Terminal:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip default-jdk
mkdir -p ~/DSBDAL
cd ~/DSBDAL
```

Copy this repository into `~/DSBDAL`, then create and activate a virtual environment:

```bash
python3 -m venv dsbdal_env
source dsbdal_env/bin/activate
pip install -r requirements.txt
```

Run a Python practical:

```bash
python3 Practical_08_Titanic_Histogram.py
```

For scripts that accept a CSV path, place the CSV in this folder and run the script with its filename, for example:

```bash
python3 Practical_05_Logistic_Regression.py Social_Network_Ads.csv
```

## 2. Datasets

- Practicals 1–3 and 6 and 10 use the Iris dataset available through scikit-learn, so they do not need a separate CSV.
- Practicals 8–9 use Seaborn's built-in Titanic dataset. The first load may require an internet connection.
- Practical 4 accepts `BostonHousing.csv` in this folder. It also attempts to fetch the Boston dataset from OpenML if the CSV is absent; this requires internet access. The Boston Housing dataset is deprecated in some teaching libraries, so use a supplied college CSV if available.
- Practical 5 requires `Social_Network_Ads.csv` in this folder. The expected columns are `Age`, `EstimatedSalary`, and `Purchased`; `Gender` is optional.
- Practical 7 uses NLTK. Its language resources may need to be downloaded once, requiring internet access.

## 3. How to create/edit and run

```bash
nano Practical_01_Data_Wrangling_1.py
```

Paste code, then press `Ctrl+O`, `Enter`, `Ctrl+X` to save and exit.

```bash
python3 Practical_01_Data_Wrangling_1.py
```

## 4. Hadoop practicals (11 and 12)

These require Hadoop to be installed and configured in local/standalone mode. Check first:

```bash
hadoop version
javac -version
```

If `hadoop` is not found, Hadoop is not configured on that computer. Do not spend exam time installing it from scratch without checking your lab setup or instructor's instructions.

### Practical 11: WordCount

From the repository root:

```bash
cd Practical_11_WordCount
mkdir -p classes
javac -classpath "$(hadoop classpath)" -d classes WordCount.java
jar -cvf wordcount.jar -C classes .
hadoop jar wordcount.jar WordCount input.txt output_wordcount
hadoop fs -cat output_wordcount/part-r-00000
```

If the output folder already exists, remove it before rerunning:

```bash
hadoop fs -rm -r output_wordcount
```

### Practical 12: Log processing

From the repository root:

```bash
cd Practical_12_LogProcessing
mkdir -p classes
javac -classpath "$(hadoop classpath)" -d classes LogProcessing.java
jar -cvf logprocessing.jar -C classes .
hadoop jar logprocessing.jar LogProcessing logs.txt output_logs
hadoop fs -cat output_logs/part-r-00000
```

If the output folder already exists:

```bash
hadoop fs -rm -r output_logs
```

## 5. Important exam notes

- Always check the exact dataset column names with `df.head()` and `df.columns`.
- CSV filenames and capitalization must match exactly.
- Python plotting programs open a graph window; close it to return to Terminal.
- Practical 4's college handout text ends mid-sentence after describing the Boston Housing dataset. This implementation uses the standard regression target `MEDV` (median house value) when available.
- Practical 12's handout only says to process a system log using MapReduce; this implementation counts log entries by log level (`INFO`, `WARNING`, `ERROR`, etc.). Confirm the expected key/value with your instructor if they gave a more specific format.
