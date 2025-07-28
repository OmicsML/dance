\[scGPT->metadatas\]+\[get_result_web->sweep_results\]+\[data_processing/merge_result_metadata.py\]->\[data/cell_type_annotation_atlas.xlsx\]
\[data/cell_type_annotation_atlas.xlsx\]+\[similarity/analyze_atlas_accuracy.py\]->\[data/in_atlas_datas\]
\[similarity/example_usage_anndata.py\]+\[data/in_atlas_datas\]+\[data/cell_type_annotation_atlas.xlsx\]->\[data/dataset_similarity\]
\[data/dataset_similarity\]+\[similarity/process_tissue_similarity_matrices.py\]->\[data/new_sim\]

#run_similarity_optimization.sh
\[data/new_sim\]+\[similarity/optimize_similarity_weights.py\]+\[cache/sweep_cache.json\]->\[data/similarity_weights_results\]
\[data/similarity_weights_results\]+\[similarity/visualize_atlas_performance.py\]+\[cache/sweep_cache.json\]->\[data/imgs\]

#注意additional sweep_id的问题
