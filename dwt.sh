#!/bin/bash

python dwt_dn.py  --noise_type_index 0
python dwt_dn.py  --noise_type_index 1
python dwt_dn.py  --noise_type_index 2
python dwt_dn.py  --noise_type_index 3

python dwt_dn.py  --noise_type_index 3 --intensity_index 0
python dwt_dn.py  --noise_type_index 3 --intensity_index 1
python dwt_dn.py  --noise_type_index 3 --intensity_index 2
python dwt_dn.py  --noise_type_index 3 --intensity_index 3
python dwt_dn.py  --noise_type_index 3 --intensity_index 4