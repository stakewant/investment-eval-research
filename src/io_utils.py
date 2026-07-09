from pathlib import Path

import pandas as pd


def read_table(path):
    """
    CSV 또는 TSV 파일을 자동으로 읽는다.
    Notion/Excel에서 내보낸 파일이 .csv 확장자여도 실제로는 탭 구분인 경우가 있어 자동 감지한다.
    """
    path = Path(path)

    encodings = ["utf-8-sig", "utf-8", "cp949"]
    last_error = None

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                sample = f.read(4096)

            tab_count = sample.count("\t")
            comma_count = sample.count(",")

            sep = "\t" if tab_count > comma_count else ","

            return pd.read_csv(
                path,
                encoding=encoding,
                sep=sep,
                engine="python"
            )

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to read table file: {path}\nLast error: {last_error}")