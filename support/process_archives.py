import pypff
import re
from datetime import datetime
import pandas as pd

print(f"[INFO] pypff version: {pypff.get_version()}")

DATA_PATH = "data\Archiv_2023_2.pst"

# Patterns to match and remove
patterns = [
    r"\bM\s\+\d{3}\d{3}\d{3}\d{3}\b",  # Phone numbers
    r"\bP\s\+\d{3}\d{3}\d{3}\d{3}\b",  # Phone numbers
    r"\S+@\S+",  # Email addresses
    r"http[s]?://\S+",  # URLs
    r"preciosalighting\.com\s*<",
    r"Facebook\s*<",  # Social Media links
    r"Instagram\s*<",
    r"Youtube\s*<",
    r"Pinterest\s*<",
    r"Linkedin\s*<",
    r"_+",  # Line of underscores
    # Czech legal disclaimer
    r"Tento e-mail je určen pouze.*od odesílatele k adresátovi\.",
    # English legal disclaimer
    r"This e-mail transmission is intended solely.*from the sender to the recipient\.",
    r"From:.*\n?",
    r"Sent:.*\n?",
    r"To:.*\n?",
    r"Cc:.*\n?",
    r"Subject:.*\n?",
    r";",  # Semicolons
    r"[^\w\s,.]",
]


def extract_emails(pst_file):
    opened_pst = pypff.open(pst_file)
    root = opened_pst.get_root_folder()

    emails = []

    def process_folder(folder):
        for folder in folder.sub_folders:
            process_folder(folder)
        for message in folder.sub_messages:
            emails.append(
                {
                    "subject": message.subject,
                    "body": message.plain_text_body,
                    "sender": message.sender_name,
                    "date": message.delivery_time,
                }
            )

    process_folder(root)
    return emails


def format_item(item, patterns):
    date = item["date"].strftime("%Y-%m-%d")
    body = item["body"].decode("utf-8")
    for pattern in patterns:
        body = re.sub(pattern, "", body)
    body = re.sub("\s+", " ", body).strip()

    return {
        "subject": item["subject"],
        "body": body,
        "sender": item["sender"],
        "date": date,
    }


def main():
    dataset_list = []
    emails = extract_emails(DATA_PATH)
    for email in emails:
        dataset_list.append(format_item(email, patterns))

    df = pd.DataFrame(dataset_list)
    df.head()
    df.to_csv("data\emails.csv", index=True, header=True, sep=";")


if __name__ == "__main__":
    main()
