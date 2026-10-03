global _EMBEDDINGS
32 if _EMBEDDINGS is None:
33 cfg = get_settings()
34 _EMBEDDINGS = HuggingFaceEmbeddings(
35 model_name=cfg.embed_model,
36 model_kwargs={"device": "cpu"},
37 encode_kwargs={"normalize_embeddings": True},
38 )
39 