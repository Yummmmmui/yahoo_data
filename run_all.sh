#!/bin/bash

echo "开始下载数据..."
python 1.yahoo_data_download.py


echo "生成视图..."
python 7.create_view.py

echo "开始质量检查..."
python 2.QC.py

echo "开始check..."
python 3.check.py

echo "正在迁移到数据库..."
python 4.migrate_to_postgres.py

echo "设置规则..."
python 5.rag_setup.py

echo "调用大模型..."
python 6.query_llama_postgres.py

