# from scrap_docs import main as scrape_docs
from transform_data import main as transform_data
from chunk_and_embed import main as chunk_and_embed




def main():
    # retrieve the needed clinical documentation
    # scrape_docs()

    # transform the retrieved documents into usable data formats
    transform_data()

    # chunk and embed the transformed data for use in retrieval-augmented generation
    chunk_and_embed()
