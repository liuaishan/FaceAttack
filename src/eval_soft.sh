python eval_soft.py \
  --test_dataset 'CASIA' \
  --test_face_path '/userhome/dataset/CASIA_align/' \
  --test_label_path '/userhome/code/FaceAttack/dataset/' \
  --target_face_path '/userhome/dataset/CASIA_align/' \
  --target_label_path '/userhome/code/FaceAttack/dataset/' \
  --model 'se_resnet_50' \
  --model_path './params_res50IR_cos_CA.pkl' \
  --model_g_path '../' 