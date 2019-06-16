python get_feature.py \
  --load_model_path  "params_res50IR_cos_CA.pkl" \
  --load_margin_path  "margin_res50IR_cos_CA.pkl" \
  --save_model_path  "params_res50IR_cos_CA.pkl"
  --save_margin_path  "margin_res50IR_cos_CA.pkl" \
  --train_data_root "CASIA_WebFace/CASIA_align/" \
  --train_data_list "CASIA_list.txt" \
  --train_data_fail "CASIA_fail.txt" \
  --train_dataset "CASIA"
  --lr 0.05 \
  --batchsize 64 \
  --epoch 6666 \
  --attack "pgd"