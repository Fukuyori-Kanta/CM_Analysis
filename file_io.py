import csv

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
    csvfile = open(file_path, 'r', encoding='utf-8')
    reader = csv.reader(csvfile)

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
    with open(dest_path, 'w') as f:
        writer = csv.writer(f)

        if len(data) >= 2: # 2次元配列以上の場合
            writer.writerows(data)

        else:   # 1次元配列の場合
            writer.writerow(data)