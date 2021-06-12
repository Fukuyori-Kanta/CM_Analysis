import os
import os.path
import configparser

def path_setting():
    """
    設定ファイル（settings.ini）から値を取得し、設定する関数
    各パスをリストにまとめて返す
    Returns
    -------
    path : list
        設定したパスのリスト[root_path, video_path, cmData_path, result_cut_path, ansData_path, result_cut_eva_path]
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
    result_cut_path = os.path.join(root_path, config['PATH']['result_cut_path']) # カット分割結果の保存ディレクトリ
    result_cut_img_path = os.path.join(root_path, config['PATH']['result_cut_img_path']) # カット画像作成結果の保存ディレクトリ

    ansData_path = os.path.join(root_path, config['PATH']['ansData_path']) # カット点の正解データのパス
    result_cut_eva_path = os.path.join(root_path, config['PATH']['result_cut_eva_path']) # カット検出評価の結果出力パス

    path = [root_path, video_path, cmData_path, result_cut_path, result_cut_img_path, ansData_path, result_cut_eva_path]

    return path