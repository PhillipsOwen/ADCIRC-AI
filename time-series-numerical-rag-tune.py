"""
Numerical RAG example using APZViz time-series station data.

"""
import sys
import faiss
import numpy as np
import pandas as pd
import os
import torch
import random

from os.path import exists
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from openai import AzureOpenAI
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer,
    TrainingArguments, DataCollatorForLanguageModeling
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# configuration parameters
# ---------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
INDEX_PATH = "numeric_rag.index"
DOCS_PATH = "numeric_docs.parquet"

# configure from env params
endpoint = os.getenv("ENDPOINT_URL", "")
deployment = os.getenv("DEPLOYMENT_NAME", "")

def get_time_series_data() -> pd.DataFrame:
    """
    example numeric dataset (time series) gathered from the APZViz UI-Data/get_station_data endpoint

    :return:
    """
    # use the save datafile if it exists
    if exists('all_station_data-full.csv'):
        print('Gathering station data over time from data file.')

        # read the CSV file with station time-series data
        df_final = pd.read_csv('all_station_data.csv', header='infer', sep=',')

        # return the data
        return df_final
    else:
        print('No data found to process.')
        sys.exit(-1)

def row_to_doc(row: pd.Series) -> str:
    """
    convert rows to text docs for embedding

    :param row:
    :return:
    """
    # keep text concise but include numeric fields and metadata
    return f"Water level observations for station {row.station} located in {row.location} on {row.datetime} was {row.Observations} ft. datetime: {row.datetime} | station: {row.station} | location: {row.location} | metric: {row.metric} | Observations: {row['Observations']} | latitude: {row['latitude']} | longitude: {row['longitude']}"

def make_train_examples(df, n_samples=200):
    examples = []
    for _ in range(n_samples):
        station = random.choice(df["station"].unique())
        window = df[df["station"] == station].sample(10)
        start = window["datetime"].min()
        end = window["datetime"].max()
        wl_mean = window["Observations"].mean()

        docs = [row_to_doc(r) for _, r in window.iterrows()]

        context = "\n".join(docs)
        question = f"What is the average water level between {start} and {end} for {station}?"
        answer = f"Avg water level: {wl_mean:.3f} m. Samples: {len(window)}."
        examples.append({
            "prompt": f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:",
            "answer": " " + answer
        })
    return examples

class NumericRAGIndex:
    """
    Class to build embeddings and the FAISS index

    """
    def __init__(self, embed_model_name=EMBED_MODEL, embed_dim=EMBED_DIM):
        """
        class initializer

        :param embed_model_name:
        :param embed_dim:
        """
        self.embedder = SentenceTransformer(embed_model_name, device=device)
        self.dim = embed_dim
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata: List[Dict[str, Any]] = []

    def build(self, df_target: pd.DataFrame):
        """
        builds the index for encoded vectors

        :param df_target:
        :return:
        """
        docs = [row_to_doc(r) for _, r in df_target.iterrows()]

        vectors = self.embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True)

        assert vectors.shape[1] == self.dim

        self.index.add(vectors)

        # keep metadata aligned with index positions
        self.metadata = [
            dict(datetime=str(r.datetime), Observations=float(str(r['Observations'])), station=r.station,
                 location=r.location, metric=r.metric, text=docs[i])
            for i, (_, r) in enumerate(df_target.iterrows())]

    def save(self, index_path=INDEX_PATH, meta_path=DOCS_PATH):
        """
        Saves the index to disk

        :param index_path:
        :param meta_path:
        :return:
        """
        faiss.write_index(self.index, index_path)
        pd.DataFrame(self.metadata).to_parquet(meta_path, index=False)

    def load(self, index_path=INDEX_PATH, meta_path=DOCS_PATH):
        """
        loads the index from disk

        :param index_path:
        :param meta_path:
        :return:
        """
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_parquet(meta_path).to_dict(orient='records')

    def query(self, q: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        queries the index

        :param q:
        :param top_k:
        :return:
        """
        # embed the query and return top_k metadata entries
        q_vec = self.embedder.encode([q], convert_to_numpy=True)
        D, I = self.index.search(q_vec, top_k)

        # init the return
        results = []

        # for each index
        for idx in I[0]:
            # is this is not a valid index
            if idx < 0 or idx >= len(self.metadata):
                continue

            # else save the result
            results.append(self.metadata[idx])

        # return the results
        return results


def compute_from_retrieved(retrieved_data: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    """
    numeric-safe retrieval with compute and a human-readable LLM explain

    :param retrieved_data:
    :param question:
    :return:
    """
    # convert the retrieved metadata into a DataFrame for safe numeric computation
    df = pd.DataFrame(retrieved_data)

    # init the output with the retrieved data
    output = {"provenance": retrieved_data}

    # simple set of common numeric ops, extend as needed.
    if "average" in question.lower() or "mean" in question.lower():
        output["computed"] = {"mean": float(df["Observations"].mean())}
    elif "max" in question.lower() or "highest" in question.lower():
        idx = df["Observations"].idxmax()
        output["computed"] = {"max": float(df.loc[idx, "Observations"]), "datetime": df.loc[idx, "datetime"]}
    elif "trend" in question.lower() or "increasing" in question.lower():
        # simple linear fit to check trend
        vals = df["Observations"].astype(float).values

        # get an array of the correct size
        x = np.arange(len(vals))

        # did we get valid data
        if len(vals) >= 2:
            # do the fitting
            m = np.polyfit(x, vals, 1)[0]

            output["computed"] = {"slope": float(m)}
        else:
            output["computed"] = {"slope": None}
    else:
        # default: return basic stats
        output["computed"] = {
            "Record count": int(df.shape[0]),
            "Observation mean": float(df["Observations"].mean()),
            "Observation std": float(df["Observations"].std()),
            "Observation min": float(df["Observations"].min()),
            "Observation max": float(df["Observations"].max())
        }

    return output


def llm_explain(computed_info: Dict[str, Any], question: str, provenance: List[Dict[str, Any]]):
    """
    Use the computed numeric result to craft a prompt for the LLM

    :param computed_info:
    :param question:
    :param provenance:
    :return:
    """

    # get the handle to the UNC Azure foundry LLM
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

    # build a concise prompt containing the computed facts and the provenance snippets.
    prov_text = "\n".join([p["text"] for p in provenance[:25]])

    # create a prompt for the output
    prompt = (
        "You are a data analyst. Only use the numeric values provided. Do NOT invent or round."
        f"Prompt: {question}\n"
        f"Verified numeric result: {computed_info}\n"
        f"Relevant data rows:\n{prov_text}\n"
        "Please write a short, clear answer that uses the computed facts and cites the date(s) from the data above."
    )

    # get the response in a human readable format
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
                                          max_tokens=1000, temperature=0.7, top_p=0.9, n=1, stream=False)

    # return the human friendly looking result
    return {"text": resp.choices[0].message.content, "computed": computed_info}

def answer_query(q, start_dt=None, end_dt=None, sensor_id=None, top_k=5):
    retrieved = retrieve(q, top_k, start_dt, end_dt, sensor_id)
    context = "\n".join(retrieved["text"].tolist())
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {q}\nANSWER:"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = peft_model.generate(ids, max_new_tokens=128, temperature=0.0)
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

def filter_by_metadata(df, start_dt=None, end_dt=None, sensor_id=None):
    filt = df
    if start_dt: filt = filt[filt["datetime"] >= start_dt]
    if end_dt: filt = filt[filt["datetime"] <= end_dt]
    if sensor_id: filt = filt[filt["sensor_id"] == sensor_id]
    return filt

def filter_data(df, start_dt=None, end_dt=None, sensor_id=None):
    filt = df.copy()
    if start_dt: filt = filt[filt["datetime"] >= start_dt]
    if end_dt: filt = filt[filt["datetime"] <= end_dt]
    if sensor_id: filt = filt[filt["sensor_id"] == sensor_id]
    return filt.reset_index(drop=True)

def retrieve(query, top_k=5, start_dt=None, end_dt=None, sensor_id=None):
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / np.linalg.norm(q_emb)
    subset = filter_by_metadata(df, start_dt, end_dt, sensor_id)
    if subset.empty: return []
    idxs = subset.index.values
    sims = (embeddings[idxs] @ q_emb).astype(float)
    top_local = np.argsort(sims)[-top_k:][::-1]
    return subset.iloc[top_local].assign(score=sims[top_local])

if __name__ == '__main__':
    """
        entry point
    """
    # load the station time-series data
    df = get_time_series_data()

    train_examples = make_train_examples(df, 400)

    base_model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    train_model = prepare_model_for_kbit_training(train_model)
    train_model = get_peft_model(train_model, LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn", "c_proj"]))
    train_model.to(device)

    dataset = Dataset.from_dict({"text": [ex["prompt"] + ex["answer"] for ex in train_examples]})
    tokenized = dataset.map(lambda e: tokenizer(e["text"], truncation=True, max_length=512), batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=train_model,
        args=TrainingArguments(
            output_dir="./csv_rag_lora",
            per_device_train_batch_size=8,
            num_train_epochs=3,
            fp16=True,
            learning_rate=2e-4,
            logging_steps=20,
        ),
        train_dataset=tokenized,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model("./csv_rag_lora_final")

    base = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(train_model, "./csv_rag_lora_final").to(device)

    # optional, if you want to save and reload it later
    # idx.save()

    # example questions humans might ask about station data
    prompts = [
        "what is the latitude and longitude of the Frying Pan Shoals location?",
        "What's the average water level in the last 3 days for the Frying Pan Shoals location?",
        "Is there an increasing trend over the last 3 days for the Frying Pan Shoals location?",
        "What were the top 3 highest values and their dates for the Frying Pan Shoals location?",
        "what is the station name for the Frying Pan Shoals location?",

        # "What's the average water level in the last 3 days for station 41013?",
        # "Is there an increasing trend over the last 3 days for station 41013?",
        # "What were the top 3 highest values and their dates for station 41013?",

        "what is the latitude and longitude of the Lockwoods Folly River location?",
        "What's the average water level in the last 3 days for the Lockwoods Folly River location?",
        "Is there an increasing trend over the last 3 days for the Lockwoods Folly River location?",
        "What were the top 3 highest Nowcast values and their dates for the Lockwoods Folly River location?",
        "what is the station name for the Lockwoods Folly River location?",

        # "What's the average water level in the last 3 days for station 30001?",
        # "Is there an increasing trend over the last 3 days for station 30001?",
        # "What were the top 3 highest values and their dates for station 30001?",
    ]

    # build index
    idx = NumericRAGIndex()
    idx.build(df)

    # output the result for each prompt
    for p in prompts:
        # get the data
        retrieved = idx.query(p, top_k=25)

        # compute the retrieved results
        computed = compute_from_retrieved(retrieved, p)

        # call the LLM using the computed facts for a human-friendly writeup
        explanation = llm_explain(computed["computed"], p, retrieved)

        print("\nPrompt:", p)
        print("Computed:", computed["computed"])

        if explanation.get("text"):
            print("LLM Answer:\n", explanation["text"])
        else:
            print("No LLM output.\n")
