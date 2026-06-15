import configparser

config = configparser.ConfigParser()
config.read(r"C:\Users\vivekhs\.aws\credentials")

key    = config["default"]["aws_access_key_id"].strip()
secret = config["default"]["aws_secret_access_key"].strip()
token  = config["default"].get("aws_session_token", "").strip()

content = f"""[aws]
aws_access_key_id = "{key}"
aws_secret_access_key = "{secret}"
aws_session_token = "{token}"
aws_region = "us-east-1"
"""

with open(r"C:\Users\vivekhs\Downloads\aws\.streamlit\secrets.toml", "w") as f:
    f.write(content)

print("secrets.toml updated successfully")
