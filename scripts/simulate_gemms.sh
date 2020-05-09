pip3 install pandas
python3 scripts/bert_gemm_simulator.py > bert_gemms_simulated.txt
python3 scripts/get_rocblas_bench_count.py bert_gemms_simulated.txt