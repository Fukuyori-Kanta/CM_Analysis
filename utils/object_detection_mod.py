import argparse
import time 
import glob
import os
import re
import csv
from file_io import create_dest_folder, read_txt, write_csv
from init_setting import setup_logger

import torch
from mmdet.apis import init_detector, inference_detector

LABEL_THRESHOLD = 0.25  # ラベル付け時の閾値

# ログ設定
logger = setup_logger(__name__)

def labeling_from_results(results, classes):
    """結果から閾値以上のラベルを返す関数

    Parameters
    ----------
    results : list
        設定ファイルのパス

    classes : list
        認識クラス一覧

    Returns
    -------
    list
        付与するラベルのリスト [[label, score], ...]
    """
    return [[classes[i], y[4]] for i, x in enumerate(results) if len(x) != 0 for _, y in enumerate(x) if y[4] >= LABEL_THRESHOLD]

def object_detection(config_file, checkpoint_file, classes_file, img_dir):
    """物体検出を行い、ラベル付け結果を返す関数

    MMDetection のAPIを用いて物体検出を行う
        参考: MMDetection(https://github.com/open-mmlab/mmdetection)

    Parameters
    ----------
    config_file : str
        Configファイルのパス

    checkpoint_file : str
        checkpointファイルのパス

    classes_file : str
        認識クラス一覧ファイル(.txt)のパス
    
    img_dir : str
        画像フォルダのパス
   
    Returns
    -------
    label_results : list
        画像ディレクトリ内の全画像のラベル付け結果
    """
    # 使用デバイスの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # モデルの初期化
    model = init_detector(config_file, checkpoint_file, device=device)

    # 認識クラスの取得
    classes = read_txt(classes_file)

    # 推論する画像パス一覧の取得
    image_files = glob.glob(os.path.join(img_dir, '**/*'))

    # 物体検出（推論）
    label_results = []  # ラベル付け結果
    for img_path in image_files:
        result = inference_detector(model, img_path)    # 推論結果
        labels = labeling_from_results(result, classes) # 付与ラベル

        video_id, file_name = img_path.replace('\\', '/').split('/')[-2:]
        cut_no = int(re.sub(r'\D', '', file_name))
        
        logger.debug(f'{video_id}, {cut_no}, {labels}')
    
        # 結果に辞書型で格納
        label_results.append({'video_id': video_id, 'cut_no': cut_no, 'labels': labels})

    return label_results

def parse_args():
    """コマンドライン引数を処理して返す関数

    Returns
    -------
    args : argparse.ArgumentParser
        解析されたコマンドライン引数
    """
    parser = argparse.ArgumentParser(description='物体検出（推論）の実行')
    parser.add_argument('config', help='Configファイルのパス')
    parser.add_argument('checkpoints', help='checkpointファイルのパス')
    parser.add_argument('classes', help='認識クラス一覧ファイル(.txt)のパス')
    parser.add_argument('img_dir', help='画像フォルダのパス')
    parser.add_argument('results_path', help='結果格納ファイル(.csv)のパス')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    start = time.time() # 開始時間
    
    # コマンドライン引数の取得
    args = parse_args()

    # ラベル付け結果の保存先フォルダの作成
    create_dest_folder(os.path.dirname(args.results_path))

    # 物体検出
    label_results = object_detection(args.config, args.checkpoints, args.classes, args.img_dir)

    # 結果をCSVファイルに保存
    field_name = ['video_id', 'cut_no', 'labels']
    with open(args.results_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_name)
        writer.writeheader()
        writer.writerows(label_results)

    # 処理時間の表示
    elapsed_time = time.time() - start
    logger.debug('elapsed_time:{0}'.format(elapsed_time) + '[sec]')