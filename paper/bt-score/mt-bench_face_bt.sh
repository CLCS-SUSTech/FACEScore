est_base_path=/data1/model/
est_models="pythia-410m-base pythia-1_4b-base llama3_1-8b-base llama3-8b-base llama3-70b-base qwen2-0_5b-base qwen2-1_5b-base qwen2-7b-base qwen2-72b-base"

for M in $est_models; do
    echo ===== ${M} =====
    echo python mt-bench_face_bt.py --est_path ${est_base_path}${M} --real
    python mt-bench_face_bt.py --est_path ${est_base_path}${M} --real
    echo python mt-bench_face_bt.py --est_path ${est_base_path}${M} --zs --real
    python mt-bench_face_bt.py --est_path ${est_base_path}${M} --zs --real
done