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
nodes=( 1 2 4 8 )
#ntaskspernode=( 4 8 16 32 )
ntaskspernode=( 128 )


parent_dir=$PWD

for g in "${nodes[@]}"
do
    echo $g
    #((ng=$nodes*$g))

    #dir="Double_layer_Dahne_ellipses_6_hypotrochoid.py/"
    # dir="Double_layer_Dahne_Result_ellipses_6/"
    #dir="Ellipses_5_hypotrochoid_test_1/"
    #dir="Double_layer_Dahne_Comparison_ellipses_5_hypotrochoid_fixed_domain_detection"
    dir="strong_scaling_5_holes/${g}"
    mkdir -p "$dir"
    cp "jobscript_strong_eigenvalues_5_holes.tmpl" "$dir/jobscript.sh"
    #cp "Dahne_ellipses_6_hypotrochoid.py" "$dir"
    #cp "Ellipses_5_hypotrochoid_test_1.py" "$dir"
    #cp "Comparison_ellipses_5_hypotrochoid_fixed_domain_detection.py" "$dir"
    cp "strong_scaling_eigenvalues_solution_5_holes.py" "$dir"
    cp "DoubleLayerWithHoles_new.py" "$dir"
    sed -i "s/_nodes_/${g}/g" "$dir/jobscript.sh"
    sed -i "s/_ntaskspernode_/${ntaskspernode}/g" "$dir/jobscript.sh"
    cd "$dir"
    mkdir -p "data"
    sbatch jobscript.sh
    cd "$parent_dir"
done
