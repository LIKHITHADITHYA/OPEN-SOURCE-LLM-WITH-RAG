import sys
import os

def log(msg):
    print(msg, flush=True)
    with open("diagnose_output.txt", "a") as f:
        f.write(msg + "\n")

if os.path.exists("diagnose_output.txt"):
    os.remove("diagnose_output.txt")

log("Testing import openai...")
from openai import OpenAI
log("openai imported successfully.")

log("Testing import chromadb...")
import chromadb
log("chromadb imported successfully.")

log("Testing import langchain_community...")
from langchain_community.utilities import SerpAPIWrapper
log("langchain_community imported successfully.")

log("Diagnostics completed.")
