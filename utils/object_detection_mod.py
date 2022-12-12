
import time
import os
import torch
from mmdet.apis import init_detector, inference_detector
import glob
import re
from file_io import create_dest_folder, read_txt, write_csv
import argparse
import csv

LABEL_THRESHOLD = 0.25  # ラベル付け時の閾値

def grant_label(result, classes):
    granted_data = []
    for i, x in enumerate(result):
        if len(x) != 0:
            for j, y in enumerate(x):
                if y[4] >= LABEL_THRESHOLD:
                    granted_data.append([classes[i], y[4]])
    return granted_data

def object_detection(config_file, checkpoint_file, classes_file, img_dir):
    # 認識クラスの取得
    classes = read_txt(classes_file)

    # モデルの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model = init_detector(config_file, checkpoint_file, device=device)

    # 推論する画像一覧
    image_files = glob.glob(os.path.join(img_dir, "**/*"))

    # 物体検出推論
    label_results = []  # ラベル付け結果
    for img_path in image_files:
        result = inference_detector(model, img_path)    # 推論結果
        labels = grant_label(result, classes)   # 付与ラベル

        video_id, file_name = img_path.replace('\\', '/').split('/')[-2:]
        cut_no = int(re.sub(r"\D", "", file_name))
        
        # 結果に辞書型で格納
        label_results.append({"video_id": video_id, "cut_no": cut_no, "labels": labels})

    return label_results

def parse_args():
    parser = argparse.ArgumentParser(
        description='物体検出実行に必要なパスを設定')
    parser.add_argument('config', help='configファイルのパス')
    parser.add_argument('checkpoints', help='checkpointファイルのパス')
    parser.add_argument('classes', help='認識クラス一覧ファイル(.txt)のパス')
    parser.add_argument('img_dir', help='画像フォルダのパス')
    parser.add_argument('results_path', help='結果格納ファイル(.csv)のパス') # 
    
    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()
    
    args = parse_args()

    # ラベル付け結果の保存先フォルダの作成
    create_dest_folder(os.path.dirname(args.results_path))

    # 物体検出
    label_results = object_detection(args.config, args.checkpoints, args.classes, args.img_dir)

    # 結果をCSVファイルに保存
    field_name = ["video_id", 'cut_no', "labels"]

    #write_csv(label_results, args.results_path)
    with open(args.results_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_name)
        writer.writeheader()
        writer.writerows(label_results)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")



