python eval.py \
  --test_dataset 'CASIA' \
  --test_face_path '/media/dsg3/datasets/CASIA_WebFace/CASIA_align/' \
  --test_label_path '/home/dsg/xuyitao/FaceAttack/dataset/' \
  --target_face_path '/media/dsg3/datasets/CASIA_WebFace/CASIA_align/' \
  --target_label_path '/home/dsg/xuyitao/FaceAttack/dataset/' \
  --model 'se_resnet_50' \
  --model_path '/media/dsg3/xuyitao/Face/model/params_res50IR_cos_CA.pkl' \
  --model_g_path '/media/dsg3/FaceAttack/model/' 