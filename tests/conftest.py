"""Shared test setup.

Tests must never send traces to LangSmith, even when your .env turns tracing on:
config.py loads .env without overriding variables that are already set, so setting
this before anything imports rag.config wins.
"""
import os

os.environ["LANGSMITH_TRACING"] = "false"
