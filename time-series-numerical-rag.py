"""
Numerical RAG example using APZViz time-series station data.

"""
import sys
import faiss
import numpy as np
import pandas as pd
import os
import torch

from os.path import exists
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from openai import AzureOpenAI

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
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def get_time_series_data() -> pd.DataFrame:
    """
    example numeric dataset (time series) gathered from the APZViz UI-Data/get_station_data endpoint

    :return:
    """
    # use the save datafile if it exists
    if exists('all_station_data.csv'):
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


if __name__ == '__main__':
    """
        entry point
    """
    # load the station time-series data
    df = get_time_series_data()

    # build index
    idx = NumericRAGIndex()
    idx.build(df)

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

        "what is the latitude and longitude of the Marcus Hook location?",
        "What's the average water level in the last 3 days for the Marcus Hook location?",
        "Is there an increasing trend over the last 3 days for the Marcus Hook location?",
        "What were the top 3 highest Nowcast values and their dates for the Marcus Hook location?",
        "what is the station name for the Marcus Hook location?",
    ]

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
