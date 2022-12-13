

import argparse
import time 
import glob
import os
import re
import csv

from file_io import create_dest_folder, read_txt, write_csv

import torch
from mmaction.apis import init_recognizer, inference_recognizer

def grant_label(results, top_n=1):
    return [[result[0], result[1]] for result in results[:top_n] if result[1] >= 0.8]

def action_recognition(config_file, checkpoint_file, classes_file, movie_dir):
    # 分類ラベルの取得
    classes = read_txt(classes_file)

    # 使用デバイスを設定(cuda:0(gpu) or cpu)
    device = 'cuda:0'
    device = torch.device(device)
   
    # モデルの初期化
    model = init_recognizer(config_file, checkpoint_file, device=device)

    # 推論する動画一覧
    movie_files = glob.glob(os.path.join(movie_dir, "**/*"))

    # 動作認識推論
    label_results = []
    for movie_path in movie_files[:30]:
        # 動画分類
        results = inference_recognizer(model, movie_path)

        # 分類結果取得
        results = [(classes[k[0]], k[1]) for k in results]
        labels = grant_label(results)

        video_id, file_name = movie_path.replace('\\', '/').split('/')[-2:]
        cut_no = int(re.sub(r"\D", "", file_name.split('.mp4')[0]))

        # 結果を辞書型で格納
        label_results.append({"video_id": video_id, "cut_no": cut_no, "labels": labels})

    return label_results

def parse_args():
    parser = argparse.ArgumentParser(
        description='動作認識実行に必要なパスを設定')
    parser.add_argument('config', help='Configファイルのパス')
    parser.add_argument('checkpoints', help='Checkpointファイルのパス')
    parser.add_argument('classes', help='認識クラス一覧ファイル(.txt)のパス')
    parser.add_argument('movie_dir', help='動画フォルダのパス')
    parser.add_argument('results_path', help='結果格納ファイル(.csv)のパス') # 
    
    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()
    
    args = parse_args()

    # ラベル付け結果の保存先フォルダの作成
    create_dest_folder(os.path.dirname(args.results_path))

    # 物体検出
    label_results = action_recognition(args.config, args.checkpoints, args.classes, args.movie_dir)

    # 結果をCSVファイルに保存
    field_name = ["video_id", 'cut_no', "labels"]

    with open(args.results_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_name)
        writer.writeheader()
        writer.writerows(label_results)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")




