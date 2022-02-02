echo "inner surface prediction"

python ./registration/test_IPNet.py --config test.conf

echo "preprocessing and scaling"

python ./registration/utils/preprocess_scan.py --config test.conf

echo "registration to SMPL+D"

python ./registration/smpl_registration/fit_SMPLD_final.py --config test.conf

echo "T-posing and smoothing"

python ./registration/smoothing.py --config test.conf

echo "registration done"