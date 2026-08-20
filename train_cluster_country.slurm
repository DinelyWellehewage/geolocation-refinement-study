for WD in 0 0.0001 0.001
do
    if [ "$WD" = "0" ]; then
        NAME="wd0"
    elif [ "$WD" = "0.0001" ]; then
        NAME="wd1e4"
    else
        NAME="wd1e3"
    fi

    echo
    echo "========================================"
    echo "STARTING WEIGHT DECAY: $WD"
    echo "========================================"

    python -m src.main_cluster_country \
        --num-clusters 64 \
        --num-countries 12 \
        --epochs 40 \
        --batch-size 64 \
        --lr 0.001 \
        --weight-decay "$WD" \
        --num-workers 4 \
        --seed 42 \
        --country-weight 0.2 \
        --smoothing-mode ordinary \
        --label-smoothing 0.1 \
        --checkpoint-name "cluster64_country_ls01_${NAME}_seed42.pth"

    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "Training failed for weight decay $WD"
        exit $STATUS
    fi

    echo "FINISHED WEIGHT DECAY: $WD"
done