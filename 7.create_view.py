# 7.create_view.py

from util.database_postgresql import create_monthly_change_view

if __name__ == '__main__':
    print("启动创建/更新月度变化视图程序")
    # 调用 database_postgresql.py 中定义的函数来执行 SQL 视图创建语句
    create_monthly_change_view()
    print("月度变化视图创建完成")