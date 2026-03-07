"""
Synthetic Dataset Generator for Fake Review Detection
Generates ~2000 reviews with realistic patterns for genuine and fake reviews.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# ── Templates ──────────────────────────────────────────────────────────────────

GENUINE_TEMPLATES = [
    "I bought this product last month and it has been working {quality} so far. {extra}",
    "The quality is {quality}. I would {recommend} this to others. {extra}",
    "After using it for a few weeks, I can say it's a {quality} product. {extra}",
    "Decent product for the price. The {feature} is {quality}. {extra}",
    "I had some issues initially but customer service helped. Overall {quality} experience. {extra}",
    "Not the best I've used but it gets the job done. {quality} value for money. {extra}",
    "This {product} met my expectations. The {feature} works as described. {extra}",
    "Ordered this for my {person} and they seem to like it. {quality} purchase overall. {extra}",
    "I've been using this for about {duration} now. It's {quality}. {extra}",
    "The packaging was neat and delivery was on time. Product quality is {quality}. {extra}",
    "Compared to other brands, this one is {quality}. {extra}",
    "I was skeptical at first but this turned out to be a {quality} buy. {extra}",
    "The {feature} could be better but overall it's a {quality} product. {extra}",
    "Works as advertised. Nothing fancy but {quality} for daily use. {extra}",
    "I read mixed reviews but decided to try it. Glad I did, it's {quality}. {extra}",
]

FAKE_TEMPLATES = [
    "AMAZING!!! Best product EVER!!! You MUST buy this RIGHT NOW!!! {hype}",
    "This is the greatest thing I have ever purchased in my entire life!!! {hype}",
    "5 stars is not enough!! This product is absolutely PERFECT in every way!! {hype}",
    "DO NOT BUY THIS! Worst product ever made! Completely terrible! {hype}",
    "I bought 10 of these because they are SO AMAZING!! Everyone needs this!! {hype}",
    "Changed my life completely!! I cannot believe how incredible this product is!! {hype}",
    "TERRIBLE TERRIBLE TERRIBLE! Do not waste your money on this garbage!! {hype}",
    "This product is a miracle! It does everything perfectly and more!! {hype}",
    "I have never been so disappointed in my life! Absolute waste of money!! {hype}",
    "BUY BUY BUY! You will not regret it! Best purchase of the year!! {hype}",
    "Absolutely phenomenal! Outstanding! Magnificent! No words can describe! {hype}",
    "Total scam! Fake product! Don't trust the seller! Horrible experience! {hype}",
    "My whole family loves this! We bought one for every room! Simply the BEST! {hype}",
    "Zero stars if I could! This ruined everything! Stay away at all costs! {hype}",
    "Incredible incredible incredible! Perfect perfect perfect! Must have! {hype}",
]

QUALITY_WORDS = ["good", "great", "decent", "okay", "solid", "fair", "nice", "fine", "reasonable", "satisfactory"]
FEATURES = ["build quality", "battery life", "screen", "design", "performance", "material", "size", "color", "sound", "texture"]
PRODUCTS = ["phone case", "charger", "headset", "mouse", "keyboard", "lamp", "blender", "backpack", "watch", "speaker"]
PERSONS = ["brother", "sister", "friend", "mom", "dad", "colleague", "partner"]
DURATIONS = ["2 weeks", "a month", "3 months", "6 months", "a year"]
RECOMMEND = ["definitely recommend", "probably recommend", "recommend with reservations", "recommend"]
EXTRAS = [
    "Would buy again.", "Happy with my purchase.", "Could be improved.", "Nothing special.",
    "Glad I chose this one.", "Does what it says.", "Pretty standard.", "No complaints.",
    "Slightly overpriced though.", "Good customer service.", "Fast shipping too.",
    "Packaging was nice.", "Matches the description.", "", "", "",
]
HYPE_PHRASES = [
    "Buy it now!!", "You won't regret it!!!", "Best ever!!!",
    "Absolutely incredible!!!", "Must have product!!!",
    "Everyone should own this!!!", "Perfect in every way!!!",
    "Greatest purchase of my life!!!", "Cannot recommend enough!!!",
    "Trust me just buy it!!!", "", "", "",
]


def generate_genuine_review():
    template = random.choice(GENUINE_TEMPLATES)
    text = template.format(
        quality=random.choice(QUALITY_WORDS),
        feature=random.choice(FEATURES),
        product=random.choice(PRODUCTS),
        person=random.choice(PERSONS),
        duration=random.choice(DURATIONS),
        recommend=random.choice(RECOMMEND),
        extra=random.choice(EXTRAS),
    )
    rating = random.choices([2, 3, 4, 5], weights=[10, 25, 40, 25])[0]
    return text.strip(), rating


def generate_fake_review():
    template = random.choice(FAKE_TEMPLATES)
    text = template.format(hype=random.choice(HYPE_PHRASES))
    rating = random.choices([1, 2, 4, 5], weights=[30, 5, 5, 60])[0]
    return text.strip(), rating


def generate_dataset(n_samples=2000, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "reviews.csv")

    records = []
    n_genuine = n_samples // 2
    n_fake = n_samples - n_genuine

    # Genuine reviews — diverse reviewers
    genuine_reviewers = [f"user_{i}" for i in range(1, n_genuine // 2 + 1)]
    base_date = datetime(2024, 1, 1)
    for i in range(n_genuine):
        text, rating = generate_genuine_review()
        reviewer = random.choice(genuine_reviewers)
        timestamp = base_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
        records.append({
            "review_text": text,
            "rating": rating,
            "label": 0,
            "reviewer_id": reviewer,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # Fake reviews — fewer reviewers (suspicious frequency)
    fake_reviewers = [f"bot_{i}" for i in range(1, 21)]
    for i in range(n_fake):
        text, rating = generate_fake_review()
        reviewer = random.choice(fake_reviewers)
        timestamp = base_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
        records.append({
            "review_text": text,
            "rating": rating,
            "label": 1,
            "reviewer_id": reviewer,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset saved to {output_path}  ({len(df)} reviews)")
    return df


if __name__ == "__main__":
    generate_dataset()
