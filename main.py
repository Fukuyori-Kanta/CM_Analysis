"""
PD3で実装したメインのプログラム

主な工程は以下の通り
    1. カット分割
    2. ラベル付け
    3. シーン統合
    4. 分析

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import csv
from logging import getLogger, StreamHandler, DEBUG

import os.path
import configparser
from cut_segmentation_mod import cut_segmentation

# ログ設定
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

def setting():
    """
    設定ファイル（settings.ini）から値を取得し、設定する関数
    パス、閾値はそれぞれリストにまとめて返す

    Returns
    -------
    path : list
        設定したパスのリスト[root_path, video_path, cmData_path, result_path]
    thresholds : list
        設定した閾値のリスト[cut_threshold, cut_between_threshold]
    """
    # --------------------------------------------------
    # 設定ファイルの読み込み
    # --------------------------------------------------
    config = configparser.ConfigParser()    
    ini_file = 'settings.ini'   # 設定ファイル

    # ファイル存在チェック
    if not os.path.exists(ini_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ini_file)
    
    config.read(ini_file, encoding='utf-8')

    # --------------------------------------------------
    # 設定ファイルから値を取得
    # --------------------------------------------------
    # パスの設定
    root_path = config['PATH']['root_path'] # ルートパス
    video_path = os.path.join(root_path, config['PATH']['video_path'])   # 動画データのパス 
    cmData_path = os.path.join(root_path, config['PATH']['cmData_path']) # CMデータのパス
    result_path = os.path.join(root_path, config['PATH']['result_path']) # 結果を格納するパス
    
    """
    # 閾値の設定
    cut_threshold = float(config['THRESHOLD']['cut_threshold'])                 # カット分割する閾値
    cut_between_threshold = float(config['THRESHOLD']['cut_between_threshold']) # カット間フレームの削除に使用する閾値
    hist_threshold = float(config['THRESHOLD']['hist_threshold'])               # ヒストグラムインタセクション（HI）の類似度に使用する閾値
    flash_threshold = float(config['THRESHOLD']['flash_threshold'])             # フラッシュ検出時のHIの類似度に使用する閾値
    effect_threshold = float(config['THRESHOLD']['effect_threshold'])           # エフェクト検出時のHIの類似度に使用する閾値
    
    thresholds = [cut_threshold, cut_between_threshold, hist_threshold, flash_threshold, effect_threshold]
    """
    path = [root_path, video_path, cmData_path, result_path]

    return path

def read_csv(file_path, needs_skip_header=False):
    """
    CSVファイルを読み込んで、その結果を返す関数

    Parameters
    ----------
    file_path : str
        読み込むCSVファイルのパス

    needs_skip_header : bool
        ヘッダーを読み飛ばすかどうか
        
    Returns
    -------
    l : list
        読み込んだ結果を返すリスト
    """
    csvfile = open(file_path, 'r', encoding='utf-8')
    reader = csv.reader(csvfile)

    # ヘッダーを読み飛ばしたい時
    if needs_skip_header:
        header = next(reader)  

    l = []
    for row in reader:
        l.append(row)
    
    return l

if __name__ == '__main__':
    # --------------------------------------------------
    # 処理開始
    # --------------------------------------------------
    logger.debug('--------------------------------------------')
    logger.debug('処理を開始します')
    logger.debug('--------------------------------------------')
    try:   
        # --------------------------------------------------
        # 各種設定
        # --------------------------------------------------
        os.chdir(r'C:\Users\hukuyori\Desktop\CM_Analysis')  # TODO 後で消す
        
        path = setting()
        root_path, video_path, cmData_path, result_path = path  # パス

        # ルートディレクトリではないとき作業ディレクトリを変更
        if os.getcwd() == root_path:
            os.chdir(root_path)
        # --------------------------------------------------
        # CMデータの読み込み
        # --------------------------------------------------
        # ファイルの存在チェック
        if not os.path.exists(cmData_path):
            raise ValueError('CSVファイルを読み込めません。')

        cm_data = read_csv(cmData_path) # CMデータ（動画ID, 企業名, 商品名, 作品名, クラスタパターン）
        video_id_list = [cm_data[i][1] for i in range(len(cm_data))]    # 動画IDリストを作成

        logger.debug('CMデータを読み込みました。')
        logger.debug('入力元 : ' + cmData_path)
        logger.debug('--------------------------------------------')

        # --------------------------------------------------
        # カット分割
        # --------------------------------------------------
        # 動画IDリストの全動画に対してカット分割
        logger.debug('カット分割を開始します。')

        cut_segmentation(video_path, result_path)   # カット分割
        
        logger.debug('全動画のカット分割が終了しました。')
        logger.debug('-------------------------------------')

        # --------------------------------------------------
        # カット画像の作成
        # --------------------------------------------------

        # --------------------------------------------------
        # 物体認識
        # --------------------------------------------------

        # --------------------------------------------------
        # 動作認識
        # --------------------------------------------------

        # --------------------------------------------------
        # シーン統合
        # --------------------------------------------------

        # --------------------------------------------------
        # 分析
        # --------------------------------------------------

    # 例外処理
    except ValueError as e:
        print(e)
    except:
        import sys
        print("Error: ", sys.exc_info()[0])
        print(sys.exc_info()[1])
        import traceback
        print(traceback.format_tb(sys.exc_info[2]))


