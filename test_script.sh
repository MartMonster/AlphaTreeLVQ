#./AlphaTree test_config.txt | tee -a test_out.txt

#./AlphaTree config/config_hybrid_64bit_8N_thr1.txt | tee -a output/out_hybrid_thr1_64bit.txt
make
./AlphaTree config/config_SENTINEL_3ch_24bit_4N_HHQ.txt | tee -a output/out_SENTINEL_3ch_serial.txt
echo $?