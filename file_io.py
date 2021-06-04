import csv
import os
import shutil

def getEncode(filepath):
    encs = "iso-2022-jp euc-jp sjis utf-8-sig".split()
    for enc in encs:
        with open(filepath, encoding=enc) as fr:
            try:
                fr = fr.read()
            except UnicodeDecodeError:
                continue
        return enc

def read_csv(file_path, needs_skip_header=False):
    """
    CSVファイルを読み込んで、その結果を返す関数

    Parameters
    ----------
    file_path : str
        読み込むCSVファイルのパス

    needs_skip_header : bool, default False
        ヘッダーを読み飛ばすかどうか
        
    Returns
    -------
    l : list
        読み込んだ結果を返すリスト
    """
    # 文字コードを取得
    enc = getEncode(file_path)

    csvfile = open(file_path, 'r', encoding=enc)
    reader = csv.reader(csvfile)

    # ヘッダーを読み飛ばしたい時
    if needs_skip_header:
        header = next(reader)  

    l = []
    for row in reader:
        l.append(row)
    
    return l

def read_favo(file_path, needs_skip_header=False):
    """
    CSVファイルを読み込んで、その結果を返す関数

    Parameters
    ----------
    file_path : str
        読み込むCSVファイルのパス

    needs_skip_header : bool, default False
        ヘッダーを読み飛ばすかどうか
        
    Returns
    -------
    l : list
        読み込んだ結果を返すリスト
    """
    # 文字コードを取得
    enc = getEncode(file_path)

    csvfile = open(file_path, 'r', encoding=enc)
    reader = csv.reader(csvfile, delimiter=',')

    # ヘッダーを読み飛ばしたい時
    if needs_skip_header:
        header = next(reader)  

    l = []
    for row in reader:
        l.append(row)
    
    return l

def write_csv(data, dest_path):
    """
    データを受け取り、CSVに書き出す関数

    Parameters
    ----------
    data : list
        出力するデータ
        
    dest_path : str
        保存先フォルダのパス
    """
    with open(dest_path, 'w', newline="", encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        if len(data) >= 2: # 2次元配列以上の場合
            writer.writerows(data)

        else:   # 1次元配列の場合
            writer.writerow(data)

def write_ranking_data(data, dest_path, i):
    """
    ランキングのデータを受け取り、CSVに書き出す関数

    Parameters
    ----------
    data : list
        出力するデータ
        
    dest_path : str
        保存先フォルダのパス
    """
    if i == 0:
        mode = 'w'
    else:
        mode = 'a'

    with open(dest_path, mode, newline="", encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        writer.writerows(data)

def dest_folder_create(dest_path):
    """
    保存先フォルダを作成する関数

    Parameters
    ----------
    dest_path : str
        保存先フォルダのパス
    """
    # 保存先フォルダの作成
    # 既に存在する場合は削除
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)    # フォルダ削除
    os.mkdir(dest_path)     # 保存先フォルダの作成