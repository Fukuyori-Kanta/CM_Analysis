import argparse
import time 
import glob
import os
import re
import csv
import operator
from file_io import create_dest_folder, read_txt, write_csv
from init_setting import setup_logger

import torch
from mmaction.apis import init_recognizer, inference_recognizer

LABEL_THRESHOLD = 0.8  # ラベル付け時の閾値

# ログ設定
logger = setup_logger(__name__)

def labeling_from_results(results, top_n=1):
    """結果から閾値以上のラベルを返す関数

    Parameters
    ----------
    results : list
        設定ファイルのパス

    top_n : int, default 1
        上位N番目までのラベルを付与するか

    Returns
    -------
    list
        付与するラベルのリスト [[label, score]]
    """
    return [[result[0], result[1]] for result in results[:top_n] if result[1] >= LABEL_THRESHOLD]

def action_recognition(config_file, checkpoint_file, classes_file, movie_dir):
    """動作認識を行い、ラベル付け結果を返す関数

    MMAction2 のAPIを用いて動作認識を行う
        参考: MMAction2(https://github.com/open-mmlab/mmaction2)

    Parameters
    ----------
    config_file : str
        Configファイルのパス

    checkpoint_file : str
        checkpointファイルのパス

    classes_file : str
        認識クラス一覧ファイル(.txt)のパス
    
    movie_dir : str
        動画フォルダのパス
   
    Returns
    -------
    label_results : list
        動画ディレクトリ内の全動画のラベル付け結果
    """
    # 使用デバイスの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
   
    # モデルの初期化
    model = init_recognizer(config_file, checkpoint_file, device=device)
    
    # 認識クラスの取得
    classes = read_txt(classes_file)

    # 推論する動画パス一覧の取得
    movie_files = glob.glob(os.path.join(movie_dir, '**/*'))

    # 動作認識（推論）
    label_results = []  # ラベル付け結果
    for movie_path in movie_files:
        results = inference_recognizer(model, movie_path)   # 推論結果
        results = [(classes[k[0]], k[1]) for k in results]
        labels = labeling_from_results(results)             # 付与ラベル

        video_id, file_name = movie_path.replace('\\', '/').split('/')[-2:]
        cut_no = int(re.sub(r'\D', '', file_name.split('.mp4')[0]))
        
        logger.debug(f'{video_id}, {cut_no}, {labels}')

        # 結果を辞書型で格納
        label_results.append({'video_id': video_id, 'cut_no': cut_no, 'labels': labels})

    return label_results

def parse_args():
    """コマンドライン引数を処理して返す関数

    Returns
    -------
    args : argparse.ArgumentParser
        解析されたコマンドライン引数
    """
    parser = argparse.ArgumentParser(description='動作認識（推論）の実行')
    parser.add_argument('config', help='Configファイルのパス')
    parser.add_argument('checkpoints', help='checkpointファイルのパス')
    parser.add_argument('classes', help='認識クラス一覧ファイル(.txt)のパス')
    parser.add_argument('movie_dir', help='動画フォルダのパス')
    parser.add_argument('results_path', help='結果格納ファイル(.csv)のパス') # 
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    start = time.time() # 開始時間
    
    # コマンドライン引数の取得
    args = parse_args()

    # ラベル付け結果の保存先フォルダの作成
    create_dest_folder(os.path.dirname(args.results_path))

    # 動作認識
    label_results = action_recognition(args.config, args.checkpoints, args.classes, args.movie_dir)
    
    # 結果をソート
    label_results = sorted(label_results, key=operator.itemgetter('video_id', 'cut_no'))
    
    # CSVファイルに保存
    field_name = ['video_id', 'cut_no', 'labels']
    with open(args.results_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_name)
        writer.writeheader()
        writer.writerows(label_results)

    # 処理時間の表示
    elapsed_time = time.time() - start
    logger.debug('elapsed_time:{0}'.format(elapsed_time) + '[sec]')