"""
Numerical RAG example using APSViz time-series station data.

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
from argparse import ArgumentParser

# ---------------------------
# configuration parameters
# ---------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# INDEX_PATH = "data/water-level/FlatIP/numeric_rag_IP.index"
# DOCS_PATH = "data/water-level/FlatIP/numeric_docs_IP.parquet"

INDEX_PATH = "data/water-level/FlatL2/numeric_rag_L2.index"
DOCS_PATH = "data/water-level/FlatL2/numeric_docs_L2.parquet"

# INDEX_PATH = "data/water-level/FlatL2/numeric_rag_L2_Normal.index"
# DOCS_PATH = "data/water-level/FlatL2/numeric_docs_L2_Normal.parquet"

# INDEX_PATH = "numeric_rag.index"
# DOCS_PATH = "numeric_docs.parquet"

llm_model_name = "gpt-4o-mini"

# configure from env params
VLLM_BASE_URL = os.getenv('ENDPOINT_URL', '')
subscription_key = os.getenv('AZURE_OPENAI_API_KEY', '')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# print('faiss version:', faiss.__version__)

def get_time_series_data() -> pd.DataFrame:
    """
    example numeric dataset (time series) gathered from the APZViz UI-Data/get_station_data endpoint

    :return:
    """
    # use the saved datafile if it exists
    if exists('all_station_data.csv'):
        print('Gathering station data over time from data file...')

        # read the CSV file with station time-series data
        df_final = pd.read_csv('all_station_data.csv', header='infer', sep=',')

        # return the data
        return df_final
    else:
        print('No data found to process. Exiting.')
        sys.exit(-1)

def row_to_doc(row: pd.Series) -> str:
    """
    convert row to a block of text of metadata to embed

    :param row:
    :return:
    """
    # keep text concise but include numeric fields and metadata
    return (f"Water level observations for station {row.station} located in {row.location} on {row.datetime} was {row.Observations}. datetime: {row.datetime} | station: {row.station} | location: {row.location} | "
            f"metric: {row.metric} | Observations: {row['Observations']} | latitude: {row['latitude']} | longitude: {row['longitude']} | "
            f"nos minor flooding level: {row['nos_minor']} | nos moderate flooding level: {row['nos_moderate']} | nos major flooding level: {row['nos_major']} | "
            f"nws minor flooding level: {row['nws_minor']} | nws moderate flooding level: {row['nws_moderate']} | nws major flooding level: {row['nws_major']}")

class NumericRAGIndex:
    """
    Class to build, search, backup and restore the FAISS vector index and metadata

    """

    def __init__(self, embed_model_name=EMBED_MODEL, embed_dim=EMBED_DIM):
        """
        class initializer

        :param embed_model_name:
        :param embed_dim:
        """
        # create a sentence embedder
        self.embedder = SentenceTransformer(embed_model_name, device=device)

        # get the dimension of the embedding
        self.dim = embed_dim

        # create a vector DB
        self.index = faiss.IndexFlatL2(self.dim)
        # self.index = faiss.IndexFlatIP(self.dim)

        # init storage for the metadata
        self.metadata: List[Dict[str, Any]] = []

    def build(self, df_target: pd.DataFrame):
        """
        builds the index for encoded vectors

        :param df_target:
        :return:
        """
        # serialize all the textual data
        docs = [row_to_doc(r) for _, r in df_target.iterrows()]

        print('Embedding vectors...')

        # create vector embeddings
        vectors = self.embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True) # Note that FlatIP DBs require normalize_embeddings=True

        # make sure we created the vectors
        assert vectors.shape[1] == self.dim

        # add the vectors
        self.index.add(vectors)

        print('Creating aligned metadata...')

        # align indexes positions and augment metadata
        self.metadata = [
            dict(datetime=str(r.datetime), Observations=float(str(r['Observations'])), station=r.station,
                 location=r.location, metric=r.metric,
                 nos_minor=r.nos_minor, nos_moderate=r.nos_moderate, nos_major=r.nos_major,
                 nws_minor=r.nws_minor, nws_moderate=r.nws_moderate, nws_major=r.nws_major,
                 flood_level=get_flood_stage(r), total_rows=df_target[df_target['location']==r.location].shape[0],
                 text=docs[i])

            for i, (_, r) in enumerate(df_target.iterrows())]

        print('Vector DB and metadata build complete.')

    def save(self, index_path=INDEX_PATH, meta_path=DOCS_PATH):
        """
        Saves the index to disk

        :param index_path:
        :param meta_path:
        :return:
        """
        print(f'Saving vector index data into {index_path}...')

        # backup the vectors
        faiss.write_index(self.index, index_path)

        print(f'Saving metadata into {meta_path}...')

        # backup the meta data
        pd.DataFrame(self.metadata).to_parquet(meta_path, index=False)

        print('Backups complete.')

    def load(self, index_path=INDEX_PATH, meta_path=DOCS_PATH):
        """
        loads the index from disk

        :param index_path:
        :param meta_path:
        :return:
        """
        print(f'Loading vector index data from {index_path}...')

        # load the vectors
        self.index = faiss.read_index(index_path)

        print(f'Loading metadata from {meta_path}...')

        # load the metadata
        self.metadata = pd.read_parquet(meta_path).to_dict(orient='records')

        print('Vector index and metadata loaded.')

    def query(self, q: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        queries the index

        :param q:
        :param top_k:
        :return:
        """
        print('Gathering vectors...')

        # embed the query and return top_k metadata entries
        q_vec = self.embedder.encode([q], convert_to_numpy=True)
        D, I = self.index.search(q_vec, top_k)

        # init the return
        results = []

        # init a record counter for score to record alignment
        counter = 0

        print('Collecting and formatting results...')

        # collect results for each index
        for idx in I[0]:
            # is this is not a valid index
            if idx < 0 or idx >= len(self.metadata):
                continue

            # add in the score
            self.metadata[idx]['score'] = float(D[0][counter])

            # save the result
            results.append(self.metadata[idx])

            # increment the score value counter
            counter += 1

        # sort the list by date
        # results = sorted(results, key=lambda x: x['score'] )

        # return the results
        return results

def get_flood_stage(values):
    """
    Uses the current water level of a station and
    gets the flood level based on the station thresholds

    """
    # init the return value
    ret_val = 'no flooding'

    if ((values['nos_major'] and values['nos_major'] - values['Observations'] < 0) or (
            values['nws_major'] and values['nws_major'] - values['Observations'] < 0)):
        ret_val = 'major flooding'
    elif ((values['nos_moderate'] and values['nos_moderate'] - values['Observations'] < 0) or (
            values['nws_moderate'] and values['nws_moderate'] - values['Observations'] < 0)):
        ret_val = 'moderate flooding'
    elif ((values['nos_minor'] and values['nos_minor'] - values['Observations'] < 0) or (
            values['nws_minor'] and values['nws_minor'] - values['Observations'] < 0)):
        ret_val = 'minor flooding'

    # print('\nStation:', values['name'], 'current_height:', values['Observations'])
    # print('values[nos_major]', values['nos_major'] - values['Observations'])
    # print('values[nos_moderate]', values['nos_moderate'] - values['Observations'])
    # print('values[nos_minor]', values['nos_minor'] - values['Observations'])
    # print('values[nws_major]', values['nws_major'] - values['Observations'])
    # print('values[nws_moderate]', values['nws_moderate'] - values['Observations'])
    # print('values[nws_minor]', values['nws_minor'] - values['Observations'])
    # print('ret_val', ret_val)

    return ret_val

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
        _idx = df["Observations"].idxmax()
        output["computed"] = {"max": float(df.loc[_idx, "Observations"]), "datetime": df.loc[_idx, "datetime"]}
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
            # "Observation mean": float(df["Observations"].mean()),
            # "Observation std": float(df["Observations"].std()),
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
        azure_endpoint=VLLM_BASE_URL,
        api_key=subscription_key,
        api_version="latest",
    )

    # build a concise prompt containing the computed facts and the provenance snippets.
    prov_text = "\n".join([p["text"] for p in provenance])

    # create a prompt for the output
    prompt = (
        "You are a data analyst. Only use the numeric values provided. Do NOT invent or round. Return nothing if there is no supporting data.\n"
        f"Prompt: {question}\n"
        f"Verified numeric result: {computed_info}\n"
        f"Relevant data rows:\n{prov_text}\n"
        "Please write a short, clear answer that uses the computed facts, cites, date(s) with latitude and longitude from the data above."
    )

    # get the response in a human readable format
    resp = client.chat.completions.create(model=llm_model_name, messages=[{"role": "user", "content": prompt}],
                                          max_tokens=1000, temperature=0.7, top_p=0.9, n=1, stream=False)

    # return the human friendly looking result
    return {"text": resp.choices[0].message.content, "computed": computed_info}


if __name__ == '__main__':
    """
        entry point
    """
    # create a command line parser
    parser = ArgumentParser()

    # add an argument for the LLM model
    parser.add_argument('--modelname', action='store', dest='modelname', default='gpt-4o-mini',
                        help='Select the Azure model name. The default is gpt-4o-mini')

    # get the args
    args = parser.parse_args()

    # assign the LLM name
    llm_model_name = args.modelname

    # create the vector index
    numeric_rag_index = NumericRAGIndex()

    # load the vector indexes and metadata
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        print('Loading saved index and metadata...')
        numeric_rag_index.load(INDEX_PATH, DOCS_PATH)
    # else load from scratch
    else:
        print('Building/creating/saving vector index DB and metadata...')

        # load the station time-series data
        df = get_time_series_data()

        # build the index
        numeric_rag_index.build(df)

        # optional, if you want to save and reload it later
        numeric_rag_index.save()

    print(f"Showing results using the {llm_model_name} LLM.")

    # example questions humans might ask about station data
    prompts = [
        "Mona island",
        # "What is happening in Fort Pulaski?",
        # "What is happening in Eastport",
        # "Should Fort Pulaski be evacuated?",
        # "What is the latitude and longitude of the Eastport station?",

        # "What is the latitude and longitude of the Marcus Hook location?",
        # "What's the average water level in the last 3 days for the Marcus Hook location?",
        # "Is there an increasing trend over the last 3 days for the Marcus Hook location?",
        # "What were the top 3 highest Nowcast values and their dates for the Marcus Hook location?",
        # "What is the station name for the Marcus Hook location?",

        # "What is the latitude and longitude of the Frying Pan Shoals location?",
        # "What's the average water level in the last 3 days for the Frying Pan Shoals location?",
        # "Is there an increasing trend over the last 3 days for the Frying Pan Shoals location?",
        # "What were the top 3 highest values and their dates for the Frying Pan Shoals location?",
        # "What is the station name for the Frying Pan Shoals location?",

        # "What's the average water level in the last 3 days for station 41013?",
        # "Is there an increasing trend over the last 3 days for station 41013?",
        # "What were the top 3 highest values and their dates for station 41013?",

        # "What is the latitude and longitude of the Lockwoods Folly River location?",
        # "What's the average water level in the last 3 days for the Lockwoods Folly River location?",
        # "Is there an increasing trend over the last 3 days for the Lockwoods Folly River location?",
        # "What were the top 3 highest Nowcast values and their dates for the Lockwoods Folly River location?",
        # "What is the station name for the Lockwoods Folly River location?",

        # "What's the average water level in the last 3 days for station 30001?",
        # "Is there an increasing trend over the last 3 days for station 30001?",
        # "What were the top 3 highest values and their dates for station 30001?",
    ]

    # output the result for each prompt
    for p in prompts:
        print("\nPrompt:", p)

        # get the data
        retrieved = numeric_rag_index.query(p, top_k=100)

        # compute the retrieved results
        computed = compute_from_retrieved(retrieved, p)

        # call the LLM using the computed facts for a human-friendly writeup
        explanation = llm_explain(computed["computed"], p, retrieved)

        print("Computed results:", computed["computed"])

        # if there was a response output it
        if explanation.get("text"):
            print(f"LLM Answer:\n{explanation['text']}")
        # alert the user of no response for this prompt
        else:
            print("No LLM output.\n")
