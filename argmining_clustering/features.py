import typing as t
from pathlib import Path

import arguebuf
import numpy as np
import spacy
from scipy.spatial.distance import cosine
from spacy.tokens.doc import Doc
from textacy.extract.keyterms.yake import yake

SPACY_MODEL = "en_core_web_lg"
parse = spacy.load(SPACY_MODEL)


# def embeddings(atoms: t.Mapping[str, arguebuf.AtomNode]) -> t.Dict[str, np.ndarray]:
#     text = {node_id: node.plain_text for node_id, node in atoms.items()}
#     response: nlp_pb2.VectorsResponse = nlp_client.Vectors(
#         nlp_pb2.VectorsRequest(
#             texts=text.values(),
#             embedding_levels=[nlp_pb2.EmbeddingLevel.EMBEDDING_LEVEL_DOCUMENT],
#             config=nlp_pb2.NlpConfig(language="en", spacy_model="en_core_web_lg"),
#         )
#     )

#     return {
#         id: np.array(adu.document.vector)
#         for id, adu in zip(text.keys(), response.vectors)
#     }


def nlp(texts: t.Iterable[str]) -> t.List[Doc]:
    return list(parse.pipe(texts))


def extract_embeddings(doc: Doc) -> np.ndarray:

    if SPACY_MODEL == "en_core_web_trf":
        return doc._.trf_data.tensors[1]

    elif SPACY_MODEL in ["en_core_web_md", "en_core_web_lg"]:
        return doc.vector  # type: ignore

    raise ValueError(f"{SPACY_MODEL} does not have vectors")


def compute_similarity_matrix(vectors: t.List[np.ndarray]) -> np.ndarray:
    """
    Logic: Compute a similarity matrix based on the embeddings of the inputs. Indexes of the results are consistens with the indexes of the original text inputs throughout the program.

    Input:
        docs - a list of spacy docs of size n created using nlp("Text")
        feature_function - selected method of feature extraction

    Output: n x n similarity matrix as numpy.ndarray



    Example:

    texts = ["A", "B", "C", "D", "E"]
    docs = [doc for doc in nlp.pipe(texts)]
    compute_similarity_matrix(docs)
    >>> array([ [1.        , 0.86532712, 0.74074012, 0.81449479, 0.79847926],
                [0.86532712, 1.        , 0.84042579, 0.83148932, 0.87582618],
                [0.74074012, 0.84042579, 1.        , 0.85297394, 0.93901908],
                [0.81449479, 0.83148932, 0.85297394, 1.        , 0.89081973],
                [0.79847926, 0.87582618, 0.93901908, 0.89081973, 1.        ] ])
    """

    n = len(vectors)
    sim_matrix = np.zeros((n, n))

    for i, vec_i in enumerate(vectors):
        for j, vec_j in enumerate(vectors):
            sim = +1 - cosine(vec_i, vec_j)
            sim_matrix[i, j] = sim

    return sim_matrix


def compute_local_keyword_similarities(dict_i, dict_j) -> np.ndarray:
    """
    Compute similaritis between all keywords of one document to the keywords of another document
    --> the local similarity measure basically computes the embedding similarity of the cross-product of keywords between two documents

    Input:
        dict_i: {keyword, embedding} of length n
        dict_j: {keyword, embedding} of length m

    Output: n x m similarity matrix as numpy.ndarray
    """

    n = len(dict_i)
    m = len(dict_j)
    sim_matrix = np.zeros((n, m))

    for i, keyword_i in enumerate(dict_i):
        for j, keyword_j in enumerate(dict_j):
            sim = +1 - cosine(dict_i[keyword_i], dict_j[keyword_j])
            sim_matrix[i, j] = sim

    # observe smilarities between named pairs
    # deff = pd.DataFrame(compute_keyword_similarity_matrix(keywords_and_spans_by_doc[0], keywords_and_spans_by_doc[1]))
    # deff.columns = [key for key in keywords_and_spans_by_doc[j].keys()]
    # deff.index = [key for key in keywords_and_spans_by_doc[i].keys()]

    return sim_matrix


def compute_global_keyword_similarity(dict_i, dict_j, cutoff=False) -> float:

    """
    Algorithm for computing keyword matching similarity (Rezaei et al.) based on the keywords of two documents
    - Find max similarity between a pair of keywords (MATCH), remove them from the pool -> Greedy


    Input:
        dict_i: {keyword, embedding} of length n
        dict_j: {keyword, embedding} of length m

    Output: similarity -> [0.0, 1.0]


    if cutoff = True, find |n-m| matches and cutoff the remaining unmatched keywords from the longer document
    if cutoff = False, remaining words are then matched to their most similar (already matched) counterpart
    """
    keyword_sim_matrix = compute_local_keyword_similarities(dict_i, dict_j)

    n, m = keyword_sim_matrix.shape

    if n < m:
        keyword_sim_matrix = keyword_sim_matrix.transpose()

    sim_matrix = keyword_sim_matrix.copy()

    rows_matched = []

    similarities = []

    while sim_matrix.any() == True:

        row_max_values = np.amax(sim_matrix, axis=1)

        row_max_values_indexes = sim_matrix.argmax(axis=1)

        i = [k for k in range(len(row_max_values_indexes))][np.argmax(row_max_values)]

        rows_matched.append(i)

        j = row_max_values_indexes[np.argmax(row_max_values)]

        sim = sim_matrix[i, j]

        similarities.append(sim)

        sim_matrix[:, j] = 0

        sim_matrix[i, :] = 0

    if cutoff == True:
        return np.sum(similarities) / (min(n, m))

    else:
        sim_matrix = keyword_sim_matrix.copy()

        for index in rows_matched:

            sim_matrix[index, :] = 0

        sim_matrix = sim_matrix[np.all(sim_matrix != 0, axis=1)]

        while sim_matrix.any() == True:

            row_max_values = np.amax(sim_matrix, axis=1)

            row_max_values_indexes = sim_matrix.argmax(axis=1)

            i = [k for k in range(len(row_max_values_indexes))][
                np.argmax(row_max_values)
            ]

            j = row_max_values_indexes[np.argmax(row_max_values)]

            sim = sim_matrix[i, j]

            similarities.append(sim)

            sim_matrix[i, :] = 0

        return np.sum(similarities) / (max(n, m))


def compute_keyword_matching_similarity_matrix(
    docs: t.List[Doc], cutoff=False
) -> np.ndarray:
    """
    Logic: Compute a >>keyword matching-based<< similarity matrix based on the embeddings of the inputs. Indexes of the results are consistens with the indexes of the original text inputs throughout the program.

    Input:
        docs - a list of spacy docs of size n created using nlp("Text")
        cutoff:
            -   if cutoff = True, find |n-m| matches and cutoff the remaining unmatched keywords from the longer document
            -   if cutoff = False, remaining words are then matched to their most similar (already matched) counterpart

    Output: n x n similarity matrix as numpy.ndarray



    Example:

    texts = ["A", "B", "C", "D", "E"]
    docs = [doc for doc in nlp.pipe(texts)]
    compute_keyword_matching_similarity_matrix(docs)
    >>> array([ [1.        , 0.86532712, 0.74074012, 0.81449479, 0.79847926],
                [0.86532712, 1.        , 0.84042579, 0.83148932, 0.87582618],
                [0.74074012, 0.84042579, 1.        , 0.85297394, 0.93901908],
                [0.81449479, 0.83148932, 0.85297394, 1.        , 0.89081973],
                [0.79847926, 0.87582618, 0.93901908, 0.89081973, 1.        ] ])
    """
    n = len(docs)

    # initialize all -1 to identify
    sim_matrix = -np.ones((n, n))

    keywords_and_vectors_by_doc = []
    for i, doc in enumerate(docs):

        keywords = [keyword for (keyword, score) in yake(doc, normalize=None)]
        keywords_and_spans = {
            keyword: (start, end)
            for (keyword, start, end) in [
                find_keyword_span(doc, keyword) for keyword in keywords
            ]
        }

        keywords_and_vectors = {
            keyword: doc[start:end].vector
            for keyword, (start, end) in keywords_and_spans.items()
        }
        keywords_and_vectors_by_doc.append(keywords_and_vectors)

    for i, dict_i in enumerate(keywords_and_vectors_by_doc):
        for j, dict_j in enumerate(keywords_and_vectors_by_doc):
            if i == j:
                sim_matrix[i, j] = 1

            else:

                # we already computed (i,j) or (j,i) for speed optimization
                if sim_matrix[i, j] != -1:
                    continue

                sim = compute_global_keyword_similarity(dict_i, dict_j, cutoff)

                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

    return sim_matrix
