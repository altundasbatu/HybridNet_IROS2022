#

##
# EDF Deterministic
nohup python benchmark/edf.py --folder=data/small_test_set --start=1 --end=200  --repeat=1 >> reproduction/edf_det_s.txt &
nohup python benchmark/edf.py --folder=data/medium_test_set --start=1 --end=200  --repeat=1 >> reproduction/edf_det_m.txt &
nohup python benchmark/edf.py --folder=data/large_test_set --start=1 --end=200  --repeat=1 >> reproduction/edf_det_l.txt &
# EDF Stochastic
nohup python benchmark/edf.py --folder=data/small_test_set --start=1 --end=200 --noise=true  --repeat=10 >> reproduction/edf_sto_s.txt &
nohup python benchmark/edf.py --folder=data/medium_test_set --start=1 --end=200 --noise=true  --repeat=10 >> reproduction/edf_sto_m.txt &
nohup python benchmark/edf.py --folder=data/large_test_set --start=1 --end=200 --noise=true --repeat=10 >> reproduction/edf_sto_l.txt &

# Genetic Deterministic
nohup python benchmark/genetic_edf.py --folder=data/small_test_set --start=1 --end=200 --generation=10 --base-population=90 --new-random=10 --new-mutation=10  --repeat=10 >> reproduction/genetic_edf_det_s.txt &
nohup python benchmark/genetic_edf.py --folder=data/medium_test_set --start=1 --end=200 --generation=10 --base-population=90 --new-random=10 --new-mutation=10  --repeat=10 >> reproduction/genetic_edf_det_m.txt &
nohup python benchmark/genetic_edf.py --folder=data/large_test_set --start=1 --end=200 --generation=10 --base-population=90 --new-random=10 --new-mutation=10  --repeat=10 >> reproduction/genetic_edf_det_l.txt &

# Genetic Stochastic
nohup python benchmark/genetic_edf.py --folder=data/small_test_set --start=1 --end=200 --generation=10 --base-population=90 --new-random=10 --new-mutation=10   --noise=true --repeat=10 >> reproduction/genetic_edf_sto_s.txt &
nohup python benchmark/genetic_edf.py --folder=data/medium_test_set --start=1 --end=200 --generation=10 --base-population=90 --new-random=10 --new-mutation=10   --noise=true --repeat=10 >> reproduction/genetic_edf_sto_m.txt &
nohup python benchmark/genetic_edf.py --folder=data/large_test_set --start=1 --end=200 --generation=10 --base-population=90 --new-random=10 --new-mutation=10   --noise=true --repeat=10 >> reproduction/genetic_edf_sto_l.txt &


## Deterministic

## HybridNet
### Small Trained
# Trained on Small, Test on Small Test Set PG 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hybridnet_small_small_pg --estimator --batch-size=8 --repeat=10 --device-id=4 >> reproduction/hnet_det_s_s_pg_8.txt &
# Trained on Small, Test on Small Test Set PG 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hybridnet_small_small_pg --estimator --batch-size=16 --repeat=10 --device-id=4 >> reproduction/hnet_det_s_s_pg_16.txt &
# Trained on Small, Test on Small Test GB 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hybridnet_small_small_pg --estimator --batch-size=8 --repeat=10 --device-id=4 >> reproduction/hnet_det_s_s_gb_8.txt &
# Trained on Small, Test on Small Test GB 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hybridnet_small_small_pg --estimator --batch-size=16 --repeat=10 --device-id=4 >> reproduction/hnet_det_s_s_gb_16.txt &

# Trained on Small, Test on Medium Test Set PG 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_small_medium_pg --estimator --batch-size=8 --repeat=10 --device-id=6 >> reproduction/hnet_det_s_m_pg_8.txt &
# Trained on Small, Test on Medium Test Set PG 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_small_medium_pg --estimator --batch-size=16 --repeat=10 --device-id=6 >> reproduction/hnet_det_s_m_pg_16.txt &
# Trained on Small, Test on Medium Test GB 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_small_medium_pg --estimator --batch-size=8 --repeat=10 --device-id=6 >> reproduction/hnet_det_s_m_gb_8.txt &
# Trained on Small, Test on Medium Test GB 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_small_medium_pg --estimator --batch-size=16 --repeat=10 --device-id=6 >> reproduction/hnet_det_s_m_gb_16.txt &

# Trained on Small, Test on Large Test Set PG 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_small_large_pg --estimator --batch-size=8 --repeat=10 --device-id=4 >> reproduction/hnet_det_s_l_pg_8.txt &
# Trained on Small, Test on Large Test Set PG 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_small_large_pg --estimator --batch-size=16 --repeat=10 --device-id=4 >> reproduction/hnet_det_s_l_pg_16.txt &
# Trained on Small, Test on Large Test GB 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_small_large_pg --estimator --batch-size=8 --repeat=10 --device-id=4 >> reproduction/hnet_det_s_l_gb_8.txt &
# Trained on Small, Test on Large Test GB 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_small_large_pg --estimator --batch-size=16 --repeat=10 --device-id=4 >> reproduction/hnet_det_s_l_gb_16.txt &

### Medium Trained
# Trained on Medium, Test on Medium Test Set PG 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_medium_medium_pg --estimator --batch-size=8 --repeat=10 --device-id=6 >> reproduction/hnet_det_m_m_pg_8.txt &
# Trained on Medium, Test on Medium Test Set PG 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_medium_medium_pg --estimator --batch-size=16 --repeat=10 --device-id=6 >> reproduction/hnet_det_m_m_pg_16.txt &
# Trained on Medium, Test on Medium Test GB 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_medium_medium_pg --estimator --batch-size=8 --repeat=10 --device-id=6 >> reproduction/hnet_det_m_m_gb_8.txt &
# Trained on Medium, Test on Medium Test GB 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_medium_medium_pg --estimator --batch-size=16 --repeat=10 --device-id=6 >> reproduction/hnet_det_m_m_gb_16.txt &

# Trained on Medium, Test on Large Test Set PG 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_medium_large_pg --estimator --batch-size=8 --repeat=10 --device-id=4 >> reproduction/hnet_det_m_l_pg_8.txt &
# Trained on Medium, Test on Large Test Set PG 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_medium_large_pg --estimator --batch-size=16 --repeat=10 --device-id=4 >> reproduction/hnet_det_m_l_pg_16.txt &
# Trained on Medium, Test on Large Test GB 8
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_medium_large_pg --estimator --batch-size=8 --repeat=10 --device-id=4 >> reproduction/hnet_det_m_l_gb_8.txt &
# Trained on Medium, Test on Large Test GB 16
nohup python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_medium_large_pg --estimator --batch-size=16 --repeat=10 --device-id=4 >> reproduction/hnet_det_m_l_gb_16.txt &

## HetGat
### Small
# Trained on Small, Test on Small Test Set PG 8
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_pg.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hetgat_small_small_pg --estimator --batch-size=8 --repeat=10 --device-id=4 >> reproduction/hgat_s_s_pg_8.txt &
# Trained on Small, Test on Small Test Set PG 16
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_pg.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hetgat_small_small_pg --estimator --batch-size=16 --repeat=10 --device-id=4 >> reproduction/hgat_s_s_pg_16.txt &
# Trained on Small, Test on Small Test GB 8
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_gb.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hetgat_small_small_pg --estimator --batch-size=8 --repeat=10 --device-id=4 >> reproduction/hgat_s_s_gb_8.txt &
# Trained on Small, Test on Small Test GB 16
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_gb.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hetgat_small_small_pg --estimator --batch-size=16 --repeat=10 --device-id=4 >> reproduction/hgat_s_s_gb_16.txt &

# Trained on Small, Test on Medium Test Set PG 8
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hetgat_small_medium_pg --estimator --batch-size=8 --repeat=10 --device-id=6 >> reproduction/hgat_s_m_pg_8.txt &
# Trained on Small, Test on Medium Test Set PG 16
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hetgat_small_medium_pg --estimator --batch-size=16 --repeat=10 --device-id=6 >> reproduction/hgat_s_m_pg_16.txt &
# Trained on Small, Test on Medium Test GB 8
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hetgat_small_medium_pg --estimator --batch-size=8 --repeat=10 --device-id=6 >> reproduction/hgat_s_m_gb_8.txt &
# Trained on Small, Test on Medium Test GB 16
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hetgat_small_medium_pg --estimator --batch-size=16 --repeat=10 --device-id=6 >> reproduction/hgat_s_m_gb_16.txt &

# Trained on Small, Test on Large Test Set PG 8
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hetgat_small_large_pg --estimator --batch-size=8 --repeat=10 --device-id=6 >> reproduction/hgat_s_l_pg_8.txt &
# Trained on Small, Test on Large Test Set PG 16
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hetgat_small_large_pg --estimator --batch-size=16 --repeat=10 --device-id=6 >> reproduction/hgat_s_l_pg_16.txt &
# Trained on Small, Test on Large Test GB 8
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hetgat_small_large_pg --estimator --batch-size=8 --repeat=10 --device-id=6 >> reproduction/hgat_s_l_gb_8.txt &
# Trained on Small, Test on Large Test GB 16
nohup python pg_eval.py --mode=best --nn=hetgat --specific-cp=data/final_trained_models/checkpoint_hetgat_small_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hetgat_small_large_pg --estimator --batch-size=16 --repeat=10 --device-id=6 >> reproduction/hgat_s_l_gb_16.txt &


# Stochastic:
## HybridNet
### Small
# Trained on Small, Test on Small Test Set PG 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_small_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_s_pg_8.txt &
# Trained on Small, Test on Small Test Set PG 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_small_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_s_pg_16.txt &
# Trained on Small, Test on Small Test GB 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_small_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_s_gb_8.txt &
# Trained on Small, Test on Small Test GB 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/small_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_small_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_s_gb_16.txt &

# Trained on Small, Test on Medium Test Set PG 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_medium_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_m_pg_8.txt &
# Trained on Small, Test on Medium Test Set PG 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_medium_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_m_pg_16.txt &
# Trained on Small, Test on Medium Test GB 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_medium_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_m_gb_8.txt &
# Trained on Small, Test on Medium Test GB 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_medium_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_m_gb_16.txt &

# Trained on Small, Test on Large Test Set PG 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_large_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_l_pg_8.txt &
# Trained on Small, Test on Large Test Set PG 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_large_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_l_pg_16.txt &
# Trained on Small, Test on Large Test GB 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_large_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_l_gb_8.txt &
# Trained on Small, Test on Large Test GB 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_small_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_stoc_small_large_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_s_l_gb_16.txt &

### Medium
# Trained on Medium, Test on Medium Test Set PG 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_stoc_medium_medium_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_m_m_pg_8.txt &
# Trained on Medium, Test on Medium Test Set PG 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_pg.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_stoc_medium_medium_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_m_m_pg_16.txt &
# Trained on Medium, Test on Medium Test GB 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_stoc_medium_medium_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_m_m_gb_8.txt &
# Trained on Medium, Test on Medium Test GB 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_gb.tar --data-folder=data/medium_test_set --save-folder=data/final_trained_models/hybridnet_stoc_medium_medium_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_m_m_gb_16.txt &

# Trained on Medium, Test on Large Test Set PG 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_stoc_medium_large_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_m_l_pg_8.txt &
# Trained on Medium, Test on Large Test Set PG 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_pg.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_stoc_medium_large_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_m_l_pg_16.txt &
# Trained on Medium, Test on Large Test GB 8
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_stoc_medium_large_pg --batch-size=8 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_m_l_gb_8.txt &
# Trained on Medium, Test on Large Test GB 16
python pg_eval.py --mode=best --nn=hybridnet --specific-cp=data/final_trained_models/checkpoint_hybridnet_medium_gb.tar --data-folder=data/large_test_set --save-folder=data/final_trained_models/hybridnet_stoc_medium_large_pg --batch-size=16 --repeat=10 --human-noise --estimator --estimator-noise --device-id=4 >> reproduction/hnet_sto_m_l_gb_16.txt &