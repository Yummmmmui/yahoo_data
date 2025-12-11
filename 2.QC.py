# coding: utf-8
import os
import pandas as pd
from redownload import main_process
from util.CSV import csv2excel
from util.time_design import time_justify
from datetime import datetime, timedelta
import calendar


def get_last_day_of_previous_month():
	# 获取当前日期
	today = datetime.today()

	# 获取当前月的第一天
	first_day_of_current_month = today.replace(day=1)

	# 获取上个月的最后一天（即当前月的第一天减去一天）
	last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)

	# 格式化为YYYYMMDD格式的字符串0
	return last_day_of_previous_month.strftime('%Y%m%d')

# 解析csv
def parse_read_csv(csv_file_path):
	# 读取csv文件的内容并解析为二维列表
	csv_lines = open(csv_file_path).readlines()
	data = []
	for line in csv_lines[1:]:
		row = line.strip().split(',')
		data.append(row)
	return data


# 清洗csv
def csv_clean(new_data, old_data):
	for i, data in enumerate(new_data):
		data[0] = data[0].replace('-', '')
		for x, Data in enumerate(data[1:5]):
			if Data != '':
				data[x+1] = str(float(Data))
		new_data[i] = data[0:5]  # 去除Adj Close和Volume两项数据

	for j, data in enumerate(old_data):
		data[0] = data[0].replace('-', '')
		for y, Data in enumerate(data[1:5]):
			if Data != '':
				data[y+1] = str(float(Data))
		old_data[j] = data[0:5]

	return new_data, old_data


# 比较新旧csv，返回差集
# 对比有的年份
def compare(new, old) -> list:
	# 获取前一个月的最后一天（当前日期的前一个月的最后一天）
	t = get_last_day_of_previous_month()

	#print(f"Previous month last day (t): {t}")  # 打印前一个月的最后一天

	# 解析读取csv文件
	new_datas = parse_read_csv(new)
	old_datas = parse_read_csv(old)

	# 数据清洗
	new_datas, old_datas = csv_clean(new_datas, old_datas)

	# 存储不同的数据行
	different = []

	for n_data in new_datas:
		# 提取日期部分并去掉时区和时间部分
		date_str = n_data[0].split()[0].replace('-', '')  # 仅获取日期部分，格式为 'YYYYMMDD'

		#print(f"Comparing row: {date_str}")  # 打印当前行的日期

		# 仅当日期不为前一个月最后一天且不在旧数据中时，才进行比较
		if n_data not in old_datas and date_str != t:
			adj_o_data = []
			adj_num = []

			# 寻找与当前行数据相同的old_datas元素
			for n, o_data in enumerate(old_datas):
				o_date_str = o_data[0].split()[0].replace('-', '')  # 提取旧数据的日期部分

				if o_date_str == date_str:
					adj_o_data.append(o_date_str)
					adj_num.append(n)

			if date_str in adj_o_data:
				num = adj_num[adj_o_data.index(date_str)]
				o_data = old_datas[num]
				# 检查数据是否为空
				if n_data[4] != '' and o_data[4] != '':
					# 判断数据是否相似
					if abs(float(n_data[4]) - float(o_data[4])) < 0.01:
						continue
				# 扩展数据行
				n_data.extend(o_data[1:5])
			# 插入文件名
			n_data.insert(0, str(new.split('\\')[-1].replace('.csv', '')))

			# 添加不同的数据行
			different.append(n_data)

	return different


# 新旧csv文件夹比较
def folder_compare(new_folder, old_folder, D_val, Redownload=False):
	"""
	比较包含CSV文件的两个文件夹，并识别数据上的差异。

    参数：
        new_folder (str)：新文件夹的路径。
        old_folder (str)：旧文件夹的路径。
        D_val (int)：识别异常数据更新的阈值。
        Redownload (bool, optional)：是否自动重新下载失败的数据。默认为False。

    返回：
        None
	"""

	print('------ Start comparing ------')

	sum = ''  # 结果报告
	total_dif = []  # 所有更新异常的数据
	name_dif = []  # 数据有更新的ticker
	redownload = ''  # 需要重新下载的文件

	# new_csv下所有文件夹
	new_files = os.listdir(new_folder)

	for new_file in new_files:
		new_path = os.path.join(new_folder, new_file)  # 新路径 new_csv/...
		n_path = os.listdir(new_path)  # xxx.csv
		old_path = os.path.join(old_folder, new_file)  # 旧路径 csv/...
		o_path = os.listdir(old_path)  # xxx.csv

		fail_record = []  # 异常记录
		record = []  # 异常记录
		common = []  # 已经下载过的文件

		print('Comparing', new_file)
		for Path in n_path:
			if Path in o_path:  # 找出之前已经下载过的文件
				common.append(Path)
			else:
				continue

			new_csv = os.path.join(new_path, Path)  # new_csv/.../xxx.csv
			old_csv = os.path.join(old_path, Path)  # csv/.../xxx.csv

			if os.path.isfile(new_csv):  # 如果此路径是一个文件
				different = compare(new_csv, old_csv)
				# 找出更新异常的
				if D_val <= len(different):  # 20变化太大
					P = Path.replace('.csv', '')
					fail_record.append(P + ' download failed.')  # 异常记录
					record.append(P)
					name_dif.append(P)  # 下载失败的ticker
					for i, di in enumerate(different):
						di.insert(0, new_file)
						different[i] = di
					total_dif.extend(different)  # 异常数据

		r = new_file + "下载失败数量: {}\n{}\n"
		redownload += '\n' + r.format(str(len(fail_record)), '\n'.join(fail_record))

		new_dif = list(set(n_path) - set(common))
		old_dif = list(set(o_path) - set(common))
		for i, dif in enumerate(new_dif):
			new_dif[i] = dif.replace('.csv', '') + '\n'
		for i, dif in enumerate(old_dif):
			old_dif[i] = dif.replace('.csv', '') + '\n'

		s = "{}\n" \
			"Number of not common files(new): {}\n" \
			"{}\n" \
			"Number of not common files(old): {}\n" \
			"{}\n" \
			"Number of different from old data: {}\n" \
			"{}\n"

		sum += s.format(new_file + ':', str(len(new_dif)), ''.join(new_dif), str(len(old_dif)), ''.join(old_dif), str(len(record)), '\n'.join(record)) + '\n'
		print(s.format('', str(len(new_dif)), '', str(len(old_dif)), '', str(len(record)), ''))

	df = pd.DataFrame(sum.split('\n'))
	df.to_csv('QC_summary.csv', index=False)

	df = pd.DataFrame(total_dif)
	df.to_csv('QC_data.csv', index=False)

	with open('failed_txt/QC_failed.txt', 'w', encoding='utf-8') as f:
		f.write(redownload)

	# 自动下载比较失败的数据
	if Redownload:
		while True:
			with open('failed_txt/QC_failed.txt', 'r', encoding='utf-8') as f:
				old_num = len(f.readlines())
			main_process(Download=True, txt='failed_txt/QC_failed.txt')
			with open('failed_txt/QC_failed.txt', 'r', encoding='utf-8') as f:
				new_num = len(f.readlines())
			# 对比下载前后数量，无更新退出循环
			if old_num == new_num:
				folder_compare(new_folder, old_folder, D_val, Redownload=False)
				break


string = 'Not comparable.\nDifference in new data:\nDate,Open,High,Low,Close\n{}\n'


# 单个csv比较
def single_file_compare(new_csv, old_csv):
	print('Comparing...')
	fail_list = []
	different = compare(new_csv, old_csv)
	for i, d in enumerate(different):
		different[i] = '  '.join(d)
	fail_list.extend(different)
	s = string.format('\n'.join(fail_list))
	return s


if __name__ == '__main__':
	'''
	# example 文件夹所有csv对比

	new_folder = r'new_csv' #新数据的文件夹
	old_folder = r'csv' #旧数据的文件夹
	D_val = 1			#两份数据的月份差（如旧数据为4月，新数据为7月，则D_val = 3）
	Redownload = False  #是否自动重新下载对比失败数据，重新下载后需改False重新对比（默认为False）
	fail_list = folder_compare(new_folder, old_folder,D_val)


	# example 单个csv对比

	new_csv = 'csv/REF-UN.TO.csv'
	old_csv = 'new_csv/REF-UN.TO.csv'
	fail_list = single_file_compare(new_csv,old_csv)

	'''
	new_folder = "new_csv"
	old_folder = "csv"
	D_val = 1
	folder_compare(new_folder, old_folder, D_val, Redownload=False)
	csv2excel()
	print('------ QC Done! ------')