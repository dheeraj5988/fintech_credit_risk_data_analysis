# MySQL Setup — Project 2 (Credit Risk)

You already have MySQL installed from Project 1. We just need a new database.

## 1. Create the new database

```bash
mysql -u root -e "CREATE DATABASE credit_risk_analytics;"
```

## 2. Verify

```bash
mysql -u root -e "SHOW DATABASES;"
```

You should see both `ecommerce_analytics` (Project 1) and `credit_risk_analytics` (Project 2) listed.

## 3. Python environment

You can reuse the same virtual environment approach as Project 1, but this
is a separate project folder, so set up a fresh venv:

```bash
cd project2_fintech_credit_risk
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Credentials

```bash
cp .env.example .env
# Edit .env — set DB_PASSWORD (same MySQL root password as Project 1)
# Set DB_NAME=credit_risk_analytics
```

That's it — you already know MySQL basics from Project 1. Onward to the data.
