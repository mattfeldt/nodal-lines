#!/bin/bash

# only change these variables

#nx=( 64 )
#ny=( 64 )
#nz=( 64 )
#Pc=( 100 )
#Nt=( 6144 )
#Nt=( 768 )
#nodes=( 1024 2048 )
#nodes=( 16 32 64 128 256 )
#nodes=( 8 )
nodes=( 1 2 4 8)
#ntaskspernode=( 4 8 16 32 )
ntaskspernode=( 128 )


parent_dir=$PWD

#for g in "${ntaskspernode[@]}"
for 
do
    echo $g
    ((ng=$nodes*$g))

    dir="Double_layer_strong_scaling_studies_MPI/ntasks_${ng}"
    mkdir -p "$dir"
    cp "jobscript_strong_eigenvalues_0_holes.tmpl" "$dir/jobscript.sh"
    cp "strong_scaling_eigenvalues_solution_0_holes.py" "$dir"
    cp "DoubleLayerWithHoles_MPI_new.py" "$dir"
    sed -i "s/_nodes_/${nodes}/g" "$dir/jobscript.sh"
    sed -i "s/_ntaskspernode_/${g}/g" "$dir/jobscript.sh"
    cd "$dir"
    mkdir -p "data"
    sbatch jobscript.sh
    cd "$parent_dir"
done
