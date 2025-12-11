# yahoo_data
该项目是一个股票数据获取、处理、存储与分析的完整解决方案，主要功能包括从 Yahoo Finance 下载多个市场的股票数据，进行数据质量控制、存储管理，并提供基于自然语言的查询与可视化分析能力。

## 功能概述

- **多市场数据获取**：支持沪深市场、标普500/罗素1000、多伦多证券交易所等市场的股票数据下载
- **双重下载机制**：结合yfinance库和直接API调用，确保数据获取稳定性
- **自动化数据处理**：自动按月度聚合数据，提取每月最后一个交易日数据
- **双存储模式**：同时支持CSV文件备份和PostgreSQL数据库存储
- **数据质量控制**：包含新旧数据对比、异常检测和完整性校验
- **智能查询系统**：通过RAG增强的LLM模型支持自然语言转SQL查询
- **可视化分析**：自动生成趋势图、排名图等可视化结果

## 项目结构
.
├── 1.yahoo_data_download.py # 股票数据下载主程序
├── 2.QC.py # 数据质量控制（新旧数据对比）
├── 3.check.py # 数据完整性检查与汇总报告
├── 4.migrate_to_postgres.py # 数据迁移至 PostgreSQL
├── 5.rag_setup.py # RAG 索引构建
├── 6.query_llama_postgres.py # 自然语言查询与可视化
├── 7.create_view.py # 创建 PostgreSQL 分析视图
├── redownload.py # 失败数据重新下载工具
├── util/ # 工具函数目录
│ ├── database_postgresql.py # PostgreSQL 数据库操作
│ └── time_design.py # 时间处理工具
├── new_csv/ # 下载的 CSV 数据（按市场分类）
├── failed_txt/ # 下载失败记录
└── chart/ # 生成的图表文件

## 环境要求
- Python 3.7+
- 依赖库：
pandas==1.5.3
yfinance==0.2.31
requests==2.31.0
openpyxl==3.1.2
psycopg2-binary==2.9.7
llama-index==0.9.45
matplotlib==3.7.1

## 注意事项
首次运行需确保master_symbol_v1.6_2023.03.xlsx文件存在（包含股票代码清单）
下载大量数据时建议使用线程模式并合理设置间隔时间，避免触发 API 限制
自然语言查询功能需要配置有效的DASHSCOPE_API_KEY
数据库存储需提前配置好 PostgreSQL 环境并创建相应用户和数据库
