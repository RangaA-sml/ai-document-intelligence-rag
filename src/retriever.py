def retrieve_documents(vector_store, query, k=3):

    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=k
    )

    for document, score in results:
        print("Similarity score:", score)

    return [
        document
        for document, score in results
    ]