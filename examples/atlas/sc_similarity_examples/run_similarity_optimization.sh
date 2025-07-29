#!/bin/bash

# Define array
array=("blood" "brain" "heart" "intestine" "kidney" "lung" "pancreas")
# Loop through array and run Python scripts
# for tissue in "${array[@]}"
# do
#     echo "Started processing tissue: $tissue"
#     python similarity/analyze_atlas_accuracy.py --tissue "$tissue" >> example_usage_anndata.log 2>&1
# done
# for tissue in "${array[@]}"
# do
#     python similarity/example_usage_anndata.py --tissue "$tissue" >> example_usage_anndata.log 2>&1
# done

# for tissue in "${array[@]}"
# do
#     # python similarity/example_usage_anndata.py --tissue "$tissue" >> example_usage_anndata.log 2>&1


#     echo "Started processing tissue: $tissue"
#     python similarity/optimize_similarity_weights.py --tissue "$tissue"
#     python visualization/visualize_atlas_performance.py --tissue "$tissue"
# done

# for tissue in "${array[@]}"
# do
#     # python similarity/example_usage_anndata.py --tissue "$tissue" >> example_usage_anndata.log 2>&1


#     python similarity/optimize_similarity_weights.py --tissue "$tissue" --in_query
#     python visualization/visualize_atlas_performance.py --tissue "$tissue" --in_query
# done


# for tissue in "${array[@]}"
# do
#     # python similarity/example_usage_anndata.py --tissue "$tissue" >> example_usage_anndata.log 2>&1


#     python similarity/optimize_similarity_weights.py --tissue "$tissue" --reduce_error
#     python visualization/visualize_atlas_performance.py --tissue "$tissue" --reduce_error

# done


# for tissue in "${array[@]}"
# do
#     # python similarity/example_usage_anndata.py --tissue "$tissue" >> example_usage_anndata.log 2>&1

#     python similarity/optimize_similarity_weights.py --tissue "$tissue" --in_query --reduce_error
#     python visualization/visualize_atlas_performance.py --tissue "$tissue" --in_query --reduce_error
# done

# # Wait for all background processes to complete
# wait

# echo "All Python scripts have completed execution"


for tissue in "${array[@]}"
do

    python visualization/visualize_atlas_performance_v2.py --tissue "$tissue"
done
