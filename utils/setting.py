import os
import os.path
import configparser
import errno

import logging

# 設定ファイル
INI_FILE = 'config/settings.ini'

def read_config(ini_file):
    """
    iniファイルを読み込み結果を返す関数

    iniファイルは以下のように設定すること

    [settings.ini]

    Parameters
    ----------
    ini_file : str
        設定ファイルのパス

    Returns
    -------
    config : configparser.ConfigParser
        iniファイルの読み込み結果
    """
    # 設定ファイルの読み込み
    config = configparser.ConfigParser()

    # ファイル存在チェック
    if not os.path.exists(ini_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ini_file)
    
    config.read(ini_file, encoding='utf-8')

    return config

def setup_logger(mod_name):
    """
    設定ファイルからログ出力パスを取得してログ設定する関数

    Parameters
    ----------
    mod_name : str
        呼び出されたモジュールの名称

    Returns
    -------
    logger : logging.Logger
        ログ設定
    """
    # --------------------------------------------------
    # 設定ファイルの読み込み
    # --------------------------------------------------
    config = read_config(INI_FILE)

    # --------------------------------------------------
    # ログの設定
    # --------------------------------------------------
    root_path = config['PATH']['root_path']                                 # ルートパス
    log_file_path = os.path.join(root_path, config['LOG']['log_file_path']) # ログ出力パス 
    log_dir = os.path.dirname(log_file_path)                                # ログディレクトリ

    # ルートディレクトリにログ保存用のディレクトリを作成
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(mod_name)
    logger.setLevel(logging.DEBUG)

    # ファイルハンドラーの作成
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
    fh.setFormatter(fh_formatter)

    # コンソールハンドラーの作成
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(ch_formatter)

    # 各ハンドラーを設定
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def setup_path():
    """
    設定ファイルからパスを取得して設定する関数
    各パスをリストにまとめて返す

    Returns
    -------
    path : list
        設定したパスのリスト
    """
    # --------------------------------------------------
    # 設定ファイルの読み込み
    # --------------------------------------------------
    config = read_config(INI_FILE)

    # --------------------------------------------------
    # 設定ファイルから値をパスを設定
    # --------------------------------------------------
    root_path = config['PATH']['root_path']                                         # ルートパス
    video_path = os.path.join(root_path, config['PATH']['video_path'])              # 動画データのパス 
    cmData_top_path = os.path.join(root_path, config['PATH']['cmData_top_path'])    # CMデータ(上位1000)のファイルパス
    cmData_btm_path = os.path.join(root_path, config['PATH']['cmData_btm_path'])    # CMデータ(下位1000)のファイルパス
    cut_path = os.path.join(root_path, config['PATH']['cut_path'])                  # カット分割結果の保存フォルダパス
    cut_img_path = os.path.join(root_path, config['PATH']['cut_img_path'])          # カット画像作成結果の保存フォルダパス
    cut_point_path = os.path.join(root_path, config['PATH']['cut_point_path'])      # カット点データ（.csv）の保存ファイルパス

    return [root_path, video_path, cmData_top_path, cmData_btm_path, cut_path, cut_img_path, cut_point_path]