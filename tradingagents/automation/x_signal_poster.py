"""
X/Twitter Signal Poster

Posts trade signals to X when trades are executed.
Requires tweepy and X API credentials in environment variables.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-load tweepy to avoid import errors if not installed
_client = None


def _get_client():
    """Get or create the tweepy Client (singleton)."""
    global _client
    if _client is not None:
        return _client

    try:
        import tweepy
    except ImportError:
        logger.warning("tweepy not installed. Run: pip install tweepy")
        return None

    consumer_key = os.getenv("X_CONSUMER_KEY")
    consumer_secret = os.getenv("X_CONSUMER_SECRET")
    access_token = os.getenv("X_ACCESS_TOKEN")
    access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")

    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        logger.warning(
            "X API credentials not configured. "
            "Set X_CONSUMER_KEY, X_CONSUMER_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET"
        )
        return None

    _client = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )
    return _client


def format_signal_tweet(
    symbol: str,
    signal: str,
    entry_price: float,
    stop_loss: float,
    take_profit: Optional[float] = None,
    rationale: Optional[str] = None,
) -> str:
    """Format a trade signal as a tweet."""
    emoji = "\U0001f7e2" if signal == "BUY" else "\U0001f534"  # green/red circle
    direction = "LONG" if signal == "BUY" else "SHORT"

    # Calculate R:R
    sl_distance = abs(entry_price - stop_loss)
    rr_str = ""
    if take_profit and sl_distance > 0:
        tp_distance = abs(take_profit - entry_price)
        rr = tp_distance / sl_distance
        rr_str = f"\nR:R {rr:.1f}"

    lines = [
        f"{emoji} #{symbol} {direction}",
        f"Entry: {entry_price}",
        f"SL: {stop_loss}",
    ]
    if take_profit:
        lines.append(f"TP: {take_profit}")
    if rr_str:
        lines.append(rr_str.strip())

    tweet = "\n".join(lines)

    # Add truncated rationale if it fits within 280 chars
    if rationale:
        # Clean up rationale - take first sentence or line
        short_rationale = rationale.split("\n")[0].split(". ")[0].strip()
        if short_rationale and len(tweet) + len(short_rationale) + 2 <= 280:
            tweet += f"\n{short_rationale}"

    return tweet[:280]


def post_trade_signal(
    symbol: str,
    signal: str,
    entry_price: float,
    stop_loss: float,
    take_profit: Optional[float] = None,
    rationale: Optional[str] = None,
) -> bool:
    """
    Post a trade signal to X/Twitter.

    Returns True if posted successfully, False otherwise.
    Never raises - failures are logged and swallowed so trading isn't affected.
    """
    try:
        client = _get_client()
        if client is None:
            return False

        tweet = format_signal_tweet(
            symbol=symbol,
            signal=signal,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rationale=rationale,
        )

        response = client.create_tweet(text=tweet)
        tweet_id = response.data["id"]
        logger.info(f"Posted signal to X: {tweet_id}")
        return True

    except Exception as e:
        logger.warning(f"Failed to post signal to X: {e}")
        return False
