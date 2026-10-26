"""Models, supplied CSVs and reproducible analysis settings."""

CONFIG = {
    "models": ["gpt-5.4", "claude-opus-4-6", "gemini-3.1-pro-preview"],
    "retrieval_replicates": 5,
    "articles_per_set": 10,
    "baselines": ["bm25_headline", "bm25_full", "dense_headline", "dense_first_chunk"],
    "retrieval_settings": {
        "gpt-5.4": {},
        "claude-opus-4-6": {"max_tokens": 4096},
        "gemini-3.1-pro-preview": {"max_tokens": 16384},
    },
    "synthesis_settings": {
        "gpt-5.4": {}, "claude-opus-4-6": {}, "gemini-3.1-pro-preview": {},
    },
    "synthesis_replicate": 0,
    "annotation_model": "gpt-5-mini",
    "annotation_replicate": 0,
    "annotation_settings": {"reasoning": {"effort": "low"}},
    "random_seed": 2026101125,
    "bootstrap_seed": 20261011,
    "bootstrap_draws": 5000,
    "frame_extraction_model": "gpt-5-mini",
    "frame_extraction_settings": {"reasoning": {"effort": "low"}},
    "frame_grouping_model": "gpt-5",
    "frame_grouping_settings": {"reasoning": {"effort": "medium"}},
    "assignment_batch_size": 50,
    "min_extraction_words": 50,
    # Match the original corpus when reusing saved experiment IDs.
    "corpus_sha256": "23871e083a6f57b5f0571e584d6c475f81d7401a1bcd0045b7a17f12aa20239a",
    # Public metadata preserves the study's IDs, query order and recorded exclusions.
    "retrieval_experiment_files": ["data_public/retrieval_experiments_public.csv"],
    "retrieval_result_files": ["data_public/retrieval_results_public.csv.gz"],
    "synthesis_experiment_files": ["data_public/synthesis_experiments_public.csv.gz"],
    "synthesis_result_files": ["data_public/synthesis_results_public.csv.gz"],
    "answer_annotation_files": ["data_public/annotated_synthesis_results_public.csv"],
    "query_files": ["data_public/queries_public.csv"],
}
