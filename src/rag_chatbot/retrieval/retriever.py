def reciprocal_rank_fusion(dense: list, sparse: list, k=60) -> list:
    rrf_dict = {}
    dense_rrf = {}
    sparse_rrf = {}
    doc_map = {}
    for i, d in enumerate(dense, start=1):
        dense_rrf[d.metadata["chunk_id"]] = 1/(k + i)
        doc_map[d.metadata["chunk_id"]] = d
    for i, d in enumerate(sparse, start=1):
        sparse_rrf[d.metadata["chunk_id"]] = 1/(k + i)
        doc_map[d.metadata["chunk_id"]] = d

    unique_documents = set(
        [d.metadata["chunk_id"] for d in dense]
        ).union(set(
            [d.metadata["chunk_id"] for d in sparse])
    )
    for d in unique_documents:
        rrf_dict[d] = 0
        if d in dense_rrf.keys():
            rrf_dict[d] += dense_rrf[d]
        if d in sparse_rrf.keys():
            rrf_dict[d] += sparse_rrf[d]
    return [doc_map[d[0]] for d in sorted(rrf_dict.items(),
                                          key=lambda x: x[1], reverse=True)]
